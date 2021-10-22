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
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import QFunction
from nnabla_rl.utils.misc import create_variables


@dataclass
class DQNQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    pass


class DQNQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: QFunction
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: DQNQTrainerConfig = DQNQTrainerConfig()):
        self._target_function = target_function
        self._prev_target_rnn_states = {}
        super(DQNQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        prev_rnn_states = self._prev_target_rnn_states
        train_rnn_states = training_variables.rnn_states
        with rnn_support(self._target_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
            max_q_value = self._target_function.max_q(s_next)
        return reward + gamma * non_terminal * max_q_value

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        rnn_states = {}
        if self._target_function.is_recurrent():
            shapes = self._target_function.internal_state_shapes()
            rnn_state_variables = create_variables(batch_size, shapes)
            rnn_states[self._target_function.scope_name] = rnn_state_variables

        training_variables.rnn_states.update(rnn_states)
        return training_variables
