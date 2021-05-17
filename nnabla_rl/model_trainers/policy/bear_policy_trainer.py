# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, QFunction, StochasticPolicy, VariationalAutoEncoder


class AdjustableLagrangeMultiplier(Model):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _log_lagrange: nn.Variable

    def __init__(self, scope_name, initial_value=None):
        super(AdjustableLagrangeMultiplier, self).__init__(
            scope_name=scope_name)
        if initial_value:
            initial_value = np.log(initial_value)
        else:
            initial_value = np.random.normal()

        initializer = np.reshape(initial_value, newshape=(1, 1))
        with nn.parameter_scope(scope_name):
            self._log_lagrange = \
                nn.parameter.get_parameter_or_create(
                    name='log_lagrange', shape=(1, 1), initializer=initializer)
        # Dummy call. Just for initializing the parameters
        self()

    def __call__(self):
        return NF.exp(self._log_lagrange)

    def clip(self, min_value, max_value):
        self._log_lagrange.d = np.clip(self._log_lagrange.d, min_value, max_value)

    @property
    def value(self):
        return np.exp(self._log_lagrange.d)


@dataclass
class BEARPolicyTrainerConfig(TrainerConfig):
    num_mmd_actions: int = 10
    mmd_sigma: float = 20.0
    mmd_type: str = 'gaussian'
    epsilon: float = 0.05
    fix_lagrange_multiplier: bool = False
    warmup_iterations: int = 20000

    def __post_init__(self):
        self._assert_one_of(self.mmd_type, ['gaussian', 'laplacian'], 'mmd_type')


class BEARPolicyTrainer(ModelTrainer):
    '''Bootstrapping Error Accumulation Reduction (BEAR) style Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: BEARPolicyTrainerConfig
    _mmd_loss: nn.Variable
    _pi_loss: nn.Variable
    _pi_warmup_loss: nn.Variable
    _lagrange_loss: nn.Variable

    _q_ensembles: Sequence[QFunction]
    _vae: VariationalAutoEncoder
    _lagrange: AdjustableLagrangeMultiplier
    _lagrange_solver: Optional[nn.solver.Solver]

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_ensembles: Sequence[QFunction],
                 vae: VariationalAutoEncoder,
                 lagrange_multiplier: AdjustableLagrangeMultiplier,
                 lagrange_solver: Optional[nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: BEARPolicyTrainerConfig = BEARPolicyTrainerConfig()):
        self._q_ensembles = q_ensembles
        self._vae = vae

        self._lagrange = lagrange_multiplier
        self._lagrange_solver = lagrange_solver
        super(BEARPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current

        # Optimize actor
        # Always forward pi loss to update the graph
        pi_loss = self._pi_warmup_loss if self._train_count < self._config.warmup_iterations else self._pi_loss
        for solver in solvers.values():
            solver.zero_grad()
        nn.forward_all([pi_loss, self._lagrange_loss])
        pi_loss.backward()
        for solver in solvers.values():
            solver.update()

        # Update lagrange_multiplier if requested
        if not self._config.fix_lagrange_multiplier:
            assert self._lagrange_solver is not None
            self._lagrange_solver.zero_grad()
            self._lagrange_loss.backward()
            self._lagrange_solver.update()
            self._lagrange.clip(-5.0, 10.0)

        trainer_state = {}
        trainer_state['pi_loss'] = float(self._pi_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)
        batch_size = training_variables.batch_size
        self._pi_loss = 0
        self._pi_warmup_loss = 0
        for policy in models:
            sampled_actions = self._vae.decode_multiple(z=None,
                                                        decode_num=self._config.num_mmd_actions,
                                                        state=training_variables.s_current)
            policy_distribution = policy.pi(training_variables.s_current)
            policy_actions = policy_distribution.sample_multiple(
                num_samples=self._config.num_mmd_actions, noise_clip=(-0.5, 0.5))

            if self._config.mmd_type == 'gaussian':
                mmd_loss = _compute_gaussian_mmd(sampled_actions, policy_actions, sigma=self._config.mmd_sigma)
            elif self._config.mmd_type == 'laplacian':
                mmd_loss = _compute_laplacian_mmd(sampled_actions, policy_actions, sigma=self._config.mmd_sigma)
            else:
                raise ValueError(
                    'Unknown mmd type: {}'.format(self._config.mmd_type))
            assert mmd_loss.shape == (batch_size, 1)

            s_hat = RF.expand_dims(training_variables.s_current, axis=0)
            s_hat = RF.repeat(s_hat, repeats=self._config.num_mmd_actions, axis=0)
            s_hat = NF.reshape(s_hat, shape=(batch_size * self._config.num_mmd_actions,
                                             training_variables.s_current.shape[-1]))
            action_shape = policy_actions.shape[-1]
            a_hat = NF.transpose(policy_actions, axes=(1, 0, 2))
            a_hat = NF.reshape(a_hat, shape=(batch_size * self._config.num_mmd_actions, action_shape))

            num_q_ensembles = len(self._q_ensembles)
            q_values = NF.stack(*(q.q(s_hat, a_hat) for q in self._q_ensembles))
            assert q_values.shape == (num_q_ensembles, self._config.num_mmd_actions * batch_size, 1)
            q_values = NF.reshape(q_values, shape=(num_q_ensembles, self._config.num_mmd_actions, batch_size, 1))
            # Compute mean among sampled actions
            q_values = NF.mean(q_values, axis=1)
            assert q_values.shape == (num_q_ensembles, batch_size, 1)

            # Compute the minimum among ensembles
            q_min = NF.min(q_values, axis=0)

            assert q_min.shape == (batch_size, 1)

            self._pi_loss += NF.mean(-q_min + self._lagrange() * mmd_loss)
            self._pi_warmup_loss += NF.mean(self._lagrange() * mmd_loss)

        # Must forward pi_loss before forwarding lagrange_loss
        self._lagrange_loss = -NF.mean(-q_min + self._lagrange() * (mmd_loss - self._config.epsilon))

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        return TrainingVariables(batch_size, s_current_var)

    def _setup_solver(self):
        super()._setup_solver()
        if not self._config.fix_lagrange_multiplier:
            self._lagrange_solver.set_parameters(self._lagrange.get_parameters(), reset=False, retain_state=True)


def _compute_gaussian_mmd(samples1, samples2, sigma):
    n = samples1.shape[1]
    m = samples2.shape[1]

    k_xx = RF.expand_dims(x=samples1, axis=2) - RF.expand_dims(x=samples1, axis=1)
    last_axis = len(k_xx.shape) - 1
    sum_k_xx = NF.sum(NF.exp(-NF.sum(k_xx**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    k_xy = RF.expand_dims(x=samples1, axis=2) - RF.expand_dims(x=samples2, axis=1)
    last_axis = len(k_xy.shape) - 1
    sum_k_xy = NF.sum(NF.exp(-NF.sum(k_xy**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    k_yy = RF.expand_dims(x=samples2, axis=2) - RF.expand_dims(x=samples2, axis=1)
    last_axis = len(k_yy.shape) - 1
    sum_k_yy = NF.sum(NF.exp(-NF.sum(k_yy**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    mmd_squared = (sum_k_xx / (n*n) - 2.0 * sum_k_xy / (m*n) + sum_k_yy / (m*m))
    # Add 1e-6 to avoid numerical instability
    return RF.sqrt(mmd_squared + 1e-6)


def _compute_laplacian_mmd(samples1, samples2, sigma):
    n = samples1.shape[1]
    m = samples2.shape[1]

    k_xx = RF.expand_dims(x=samples1, axis=2) - RF.expand_dims(x=samples1, axis=1)
    last_axis = len(k_xx.shape) - 1
    sum_k_xx = NF.sum(NF.exp(-NF.sum(NF.abs(k_xx), axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    k_xy = RF.expand_dims(x=samples1, axis=2) - RF.expand_dims(x=samples2, axis=1)
    last_axis = len(k_xy.shape) - 1
    sum_k_xy = NF.sum(NF.exp(-NF.sum(NF.abs(k_xy), axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    k_yy = RF.expand_dims(x=samples2, axis=2) - RF.expand_dims(x=samples2, axis=1)
    last_axis = len(k_yy.shape) - 1
    sum_k_yy = NF.sum(NF.exp(-NF.sum(NF.abs(k_yy), axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

    mmd_squared = (sum_k_xx / (n*n) - 2.0 * sum_k_xy / (m*n) + sum_k_yy / (m*m))
    # Add 1e-6 to avoid numerical instability
    return RF.sqrt(mmd_squared + 1e-6)
