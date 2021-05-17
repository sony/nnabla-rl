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
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, StochasticPolicy


@dataclass
class SPGPolicyTrainerConfig(TrainerConfig):
    pi_loss_scalar: float = 1.0
    grad_clip_norm: Optional[float] = None


class SPGPolicyTrainer(ModelTrainer):
    '''Stochastic Policy Gradient (SPG) style Policy Trainer
    Stochastic Policy Gradient is widely known as 'Policy Gradient algorithm'
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: SPGPolicyTrainerConfig
    _pi_loss: nn.Variable
    _target_return: nn.Variable

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SPGPolicyTrainerConfig = SPGPolicyTrainerConfig()):
        super(SPGPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def setup_batch(self, batch: TrainingBatch):
        target_return = batch.extra['target_return']
        prev_batch_size = self._target_return.shape[0]
        new_batch_size = target_return.shape[0]
        if prev_batch_size != new_batch_size:
            self._target_return = nn.Variable((new_batch_size, 1))
        target_return = np.reshape(target_return, self._target_return.shape)
        self._target_return.d = target_return
        return batch

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)

        for solver in solvers.values():
            if self._config.grad_clip_norm is not None:
                solver.clip_grad_by_norm(self._config.grad_clip_norm)
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = float(self._pi_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)

        # Actor optimization graph
        target_value = self._compute_target(training_variables)
        target_value.need_grad = False

        self._pi_loss = 0
        for policy in models:
            self._pi_loss += self._compute_loss(policy, target_value, training_variables)

    def _compute_loss(self,
                      model: StochasticPolicy,
                      target_value: nn.Variable,
                      training_variables: TrainingVariables) -> nn.Variable:
        distribution = model.pi(training_variables.s_current)
        log_prob = distribution.log_prob(training_variables.a_current)
        return NF.sum(-log_prob * target_value) * self._config.pi_loss_scalar

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        raise NotImplementedError

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            action_shape = (batch_size, 1)
        else:
            action_shape = (batch_size, self._env_info.action_dim)
        a_current_var = nn.Variable(action_shape)
        return TrainingVariables(batch_size, s_current_var, a_current_var)
