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
from typing import Dict

import nnabla as nn
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import QFunction


@dataclass
class DDQNQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    pass


class DDQNQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _train_function: QFunction
    _target_function: QFunction

    def __init__(self,
                 train_function: QFunction,
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: DDQNQTrainerConfig = DDQNQTrainerConfig()):
        self._train_function = train_function
        self._target_function = target_function
        super(DDQNQTrainer, self).__init__(train_function, solvers, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next
        assert gamma is not None
        assert reward is not None
        assert non_terminal is not None
        assert s_next is not None

        a_next = self._train_function.argmax_q(s_next)
        double_q_target = self._target_function.q(s_next, a_next)
        return reward + gamma * non_terminal * double_q_target
