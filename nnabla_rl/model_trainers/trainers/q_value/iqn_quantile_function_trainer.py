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

from typing import cast, Dict, Sequence

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import \
    TrainerConfig, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import StateActionQuantileFunction, Model


@dataclass
class IQNQuantileFunctionTrainerConfig(TrainerConfig):
    N: int = 64
    N_prime: int = 64
    K: int = 32
    kappa: float = 1.0


class IQNQuantileFunctionTrainer(ModelTrainer):
    _config: IQNQuantileFunctionTrainerConfig
    _quantile_huber_loss: nn.Variable

    def __init__(self,
                 env_info: EnvironmentInfo,
                 config: IQNQuantileFunctionTrainerConfig = IQNQuantileFunctionTrainerConfig()):
        super(IQNQuantileFunctionTrainer, self).__init__(env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current
        training_variables.reward.d = batch.reward
        training_variables.gamma.d = batch.gamma
        training_variables.non_terminal.d = batch.non_terminal
        training_variables.s_next.d = batch.s_next

        for solver in solvers.values():
            solver.zero_grad()
        self._quantile_huber_loss.forward()
        self._quantile_huber_loss.backward()
        for solver in solvers.values():
            solver.update()

        # TODO: return dictionary of computed errors
        return {}

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        models = cast(Sequence[StateActionQuantileFunction], models)

        kwargs = {}
        kwargs['K'] = self._config.K
        kwargs['N_prime'] = self._config.N_prime
        batch_size = training_variables.batch_size

        target = training.compute_target(training_variables, **kwargs)
        target = RF.expand_dims(target, axis=1)
        target.need_grad = False
        assert target.shape == (batch_size, 1, self._config.N_prime)

        self._quantile_huber_loss = 0
        for model in models:
            tau_i = model._sample_tau(shape=(batch_size, self._config.N))
            samples = model.return_samples(training_variables.s_current, tau_i)
            Z_tau_i = model._return_samples_of(samples, training_variables.a_current)
            Z_tau_i = RF.expand_dims(Z_tau_i, axis=2)
            tau_i = RF.expand_dims(tau_i, axis=2)
            assert Z_tau_i.shape == (batch_size, self._config.N, 1)
            assert tau_i.shape == Z_tau_i.shape

            quantile_huber_loss = RF.quantile_huber_loss(target, Z_tau_i, self._config.kappa, tau_i)
            assert quantile_huber_loss.shape == (batch_size, self._config.N, self._config.N_prime)
            quantile_huber_loss = NF.mean(quantile_huber_loss, axis=2)
            quantile_huber_loss = NF.sum(quantile_huber_loss, axis=1)
            self._quantile_huber_loss += NF.mean(quantile_huber_loss)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        a_current_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))
        reward_var = nn.Variable((batch_size, 1))
        gamma_var = nn.Variable((1, 1))
        non_terminal_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))

        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               a_current=a_current_var,
                                               reward=reward_var,
                                               gamma=gamma_var,
                                               non_terminal=non_terminal_var,
                                               s_next=s_next_var)
        return training_variables
