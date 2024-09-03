# Copyright 2024 Sony Group Corporation.
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
from typing import Dict, Sequence, Union

import numpy as np

import nnabla as nn
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.v_value.v_function_trainer import VFunctionTrainer, VFunctionTrainerConfig
from nnabla_rl.models import Model, QFunction, VFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class IQLVFunctionTrainerConfig(VFunctionTrainerConfig):
    expectile: float = 0.7


class IQLVFunctionTrainer(VFunctionTrainer):
    _config: IQLVFunctionTrainerConfig

    def __init__(
        self,
        models: Union[VFunction, Sequence[VFunction]],
        solvers: Dict[str, nn.solver.Solver],
        target_functions: Union[QFunction, Sequence[QFunction]],
        env_info: EnvironmentInfo,
        config: IQLVFunctionTrainerConfig = IQLVFunctionTrainerConfig(),
    ):
        self._target_functions = convert_to_list_if_not_list(target_functions)
        super().__init__(models, solvers, env_info, config)

    def _update_model(
        self,
        models: Sequence[Model],
        solvers: Dict[str, nn.solver.Solver],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.a_current, b.a_current)

        return super()._update_model(models, solvers, batch, training_variables, **kwargs)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs):
        s_current = training_variables.s_current
        a_current = training_variables.a_current
        q_values = []
        for q_function in self._target_functions:
            q_values.append(q_function.q(s_current, a_current))
        target_q = RNF.minimum_n(q_values)
        return target_q

    def _compute_loss(
        self, model: VFunction, target_value: nn.Variable, training_variables: TrainingVariables
    ) -> nn.Variable:
        v_value = model.v(training_variables.s_current)
        td_error = target_value - v_value
        return RNF.expectile_regression(td_error, self._config.expectile)

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)
        training_variables.a_current = create_variable(batch_size, self._env_info.action_shape)

        return training_variables
