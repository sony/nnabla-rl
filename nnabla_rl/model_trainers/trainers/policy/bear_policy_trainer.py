from typing import Iterable, Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

import nnabla_rl.functions as RF
from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import Model, StochasticPolicy, QFunction, VariationalAutoEncoder


class AdjustableLagrangeMultiplier(Model):
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
class BEARPolicyTrainerParam(TrainerParam):
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

    def __init__(self,
                 env_info,
                 q_ensembles: Iterable[QFunction],
                 vae: VariationalAutoEncoder,
                 lagrange_multiplier: AdjustableLagrangeMultiplier,
                 lagrange_solver: nn.solver.Solver = None,
                 params: BEARPolicyTrainerParam = BEARPolicyTrainerParam()):
        super(BEARPolicyTrainer, self).__init__(env_info, params)
        self._q_ensembles = q_ensembles
        self._vae = vae

        self._lagrange = lagrange_multiplier
        self._lagrange_solver = lagrange_solver

        self._mmd_loss = None
        self._pi_loss = None
        self._pi_warmup_loss = None
        self._lagrange_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs):
        training_variables.s_current.d = batch.s_current

        # Optimize actor
        # Always forward pi loss to update the graph
        pi_loss = self._pi_warmup_loss if self._train_count < self._params.warmup_iterations else self._pi_loss
        for solver in solvers.values():
            solver.zero_grad()
        nn.forward_all([pi_loss, self._lagrange_loss])
        pi_loss.backward()
        for solver in solvers.values():
            solver.update()

        # Update lagrange_multiplier if requested
        if not self._params.fix_lagrange_multiplier:
            self._lagrange_solver.zero_grad()
            self._lagrange_loss.backward()
            self._lagrange_solver.update()
            self._lagrange.clip(-5.0, 10.0)

        errors = {}
        return errors

    def _build_training_graph(self, models: Iterable[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        if not isinstance(models[0], StochasticPolicy):
            raise ValueError

        batch_size = training_variables.batch_size
        self._pi_loss = 0
        self._pi_warmup_loss = 0
        for policy in models:
            sampled_actions = self._vae.decode_multiple(
                self._params.num_mmd_actions, training_variables.s_current)
            policy_distribution = policy.pi(training_variables.s_current)
            policy_actions = policy_distribution.sample_multiple(
                num_samples=self._params.num_mmd_actions, noise_clip=(-0.5, 0.5))

            if self._params.mmd_type == 'gaussian':
                mmd_loss = _compute_gaussian_mmd(sampled_actions, policy_actions, sigma=self._params.mmd_sigma)
            elif self._params.mmd_type == 'laplacian':
                mmd_loss = _compute_laplacian_mmd(sampled_actions, policy_actions, sigma=self._params.mmd_sigma)
            else:
                raise ValueError(
                    'Unknown mmd type: {}'.format(self._params.mmd_type))
            assert mmd_loss.shape == (batch_size, 1)

            s_hat = RF.expand_dims(training_variables.s_current, axis=0)
            s_hat = RF.repeat(s_hat, repeats=self._params.num_mmd_actions, axis=0)
            s_hat = NF.reshape(s_hat, shape=(batch_size * self._params.num_mmd_actions,
                                             training_variables.s_current.shape[-1]))
            action_shape = policy_actions.shape[-1]
            a_hat = NF.transpose(policy_actions, axes=(1, 0, 2))
            a_hat = NF.reshape(a_hat, shape=(batch_size * self._params.num_mmd_actions, action_shape))

            num_q_ensembles = len(self._q_ensembles)
            q_values = NF.stack(*(q.q(s_hat, a_hat) for q in self._q_ensembles))
            assert q_values.shape == (num_q_ensembles, self._params.num_mmd_actions * batch_size, 1)
            q_values = NF.reshape(q_values, shape=(num_q_ensembles, self._params.num_mmd_actions, batch_size, 1))
            # Compute mean among sampled actions
            q_values = NF.mean(q_values, axis=1)
            assert q_values.shape == (num_q_ensembles, batch_size, 1)

            # Compute the minimum among ensembles
            q_min = NF.min(q_values, axis=0)

            assert q_min.shape == (batch_size, 1)

            self._pi_loss += NF.mean(-q_min + self._lagrange() * mmd_loss)
            self._pi_warmup_loss += NF.mean(self._lagrange() * mmd_loss)

        # Must forward pi_loss before forwarding lagrange_loss
        self._lagrange_loss = -NF.mean(-q_min + self._lagrange() * (mmd_loss - self._params.epsilon))

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        return TrainingVariables(batch_size, s_current_var)

    def _setup_solver(self):
        super()._setup_solver()
        if not self._params.fix_lagrange_multiplier:
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
