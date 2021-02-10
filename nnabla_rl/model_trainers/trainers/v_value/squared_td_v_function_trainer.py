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

from typing import cast, Dict, Iterable

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import \
    TrainerConfig, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import VFunction, Model


@dataclass
class SquaredTDVFunctionTrainerConfig(TrainerConfig):
    reduction_method: str = 'mean'
    v_loss_scalar: float = 1.0

    def __post_init__(self):
        super(SquaredTDVFunctionTrainerConfig, self).__post_init__()
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')


class SquaredTDVFunctionTrainer(ModelTrainer):
    _config: SquaredTDVFunctionTrainerConfig
    _v_loss: nn.Variable  # Training loss/output

    def __init__(self,
                 env_info: EnvironmentInfo,
                 config: SquaredTDVFunctionTrainerConfig = SquaredTDVFunctionTrainerConfig()):
        super(SquaredTDVFunctionTrainer, self).__init__(env_info, config)

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._v_loss.forward(clear_no_need_grad=True)
        self._v_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        return {}

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        models = cast(Iterable[VFunction], models)

        # value function learning
        target_v = training.compute_target(training_variables)

        td_errors = [target_v - v_function.v(training_variables.s_current) for v_function in models]
        v_loss = 0
        for td_error in td_errors:
            squared_td_error = NF.pow_scalar(td_error, 2.0)
            if self._config.reduction_method == 'mean':
                v_loss += self._config.v_loss_scalar * NF.mean(squared_td_error)
            elif self._config.reduction_method == 'sum':
                v_loss += self._config.v_loss_scalar * NF.sum(squared_td_error)
            else:
                raise RuntimeError
        self._v_loss = v_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        training_variables = TrainingVariables(batch_size, s_current_var)
        return training_variables
