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
from typing import Dict, Sequence, Union

import nnabla as nn
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import QFunction


@dataclass
class ClippedDoubleQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    pass


class ClippedDoubleQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Sequence[QFunction],
                 env_info: EnvironmentInfo,
                 config: ClippedDoubleQTrainerConfig = ClippedDoubleQTrainerConfig()):
        if len(target_functions) < 2:
            raise ValueError('Must have at least 2 target functions for training')
        self._target_functions = target_functions
        super(ClippedDoubleQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        q_values = []
        a_next = self._target_functions[0].argmax_q(s_next)
        for target_function in self._target_functions:
            q_value = target_function.q(s_next, a_next)
            q_values.append(q_value)
        target_q = RNF.minimum_n(q_values)
        return reward + gamma * non_terminal * target_q
