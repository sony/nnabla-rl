# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from typing import cast, Dict, Optional, Sequence

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import Model, StochasticPolicy


@dataclass
class SPGPolicyTrainerParam(TrainerParam):
    pi_loss_scalar: float = 1.0
    grad_clip_norm: Optional[float] = None


class SPGPolicyTrainer(ModelTrainer):
    '''Stochastic Policy Gradient (SPG) style Policy Trainer
    Stochastic Policy Gradient is widely known as 'Policy Gradient algorithm'
    '''

    _params: SPGPolicyTrainerParam
    _pi_loss: nn.Variable

    def __init__(self,
                 env_info: EnvironmentInfo,
                 params: SPGPolicyTrainerParam = SPGPolicyTrainerParam()):
        super(SPGPolicyTrainer, self).__init__(env_info, params)

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
            if self._params.grad_clip_norm is not None:
                solver.clip_grad_by_norm(self._params.grad_clip_norm)
            solver.update()

        return {}

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)

        # Actor optimization graph
        target_value = training.compute_target(training_variables)

        self._pi_loss = 0
        for policy in models:
            distribution = policy.pi(training_variables.s_current)
            log_prob = distribution.log_prob(training_variables.a_current)
            self._pi_loss += NF.sum(-log_prob * target_value) * self._params.pi_loss_scalar

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            action_shape = (batch_size, 1)
        else:
            action_shape = (batch_size, self._env_info.action_dim)
        a_current_var = nn.Variable(action_shape)
        return TrainingVariables(batch_size, s_current_var, a_current_var)
