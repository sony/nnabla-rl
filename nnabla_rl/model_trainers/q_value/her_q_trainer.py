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
from typing import Dict, Optional, Sequence, Tuple, Union

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import DeterministicPolicy, QFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list
from nnabla_rl.utils.misc import create_variables


@dataclass
class HERQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    return_clip: Optional[Tuple[float, float]] = None


class HERQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: HERQTrainerConfig
    _target_functions: Sequence[QFunction]
    _target_policy: DeterministicPolicy
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[QFunction, Sequence[QFunction]],
                 target_policy: DeterministicPolicy,
                 env_info: EnvironmentInfo,
                 config: HERQTrainerConfig = HERQTrainerConfig()):
        self._target_policy = target_policy
        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._prev_target_rnn_states = {}
        self._prev_q_rnn_states = {}
        super(HERQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        train_rnn_states = training_variables.rnn_states
        prev_rnn_states = self._prev_target_rnn_states
        with rnn_support(self._target_policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
            a_next = self._target_policy.pi(s_next)
        a_next.need_grad = False

        q_values = []
        prev_rnn_states = self._prev_q_rnn_states
        for target_q_function in self._target_functions:
            with rnn_support(target_q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_value = target_q_function.q(s_next, a_next)
                q_values.append(q_value)

        # Use the minimum among computed q_values by default
        target_q = RNF.minimum_n(q_values)
        return_value = reward + gamma * non_terminal * target_q
        if self._config.return_clip is not None:
            minimum = self._config.return_clip[0]
            maximum = self._config.return_clip[1]
            return_value = NF.clip_by_value(return_value, minimum, maximum)
        return return_value

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        rnn_states = {}
        for target_function in self._target_functions:
            if target_function.is_recurrent():
                shapes = target_function.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[target_function.scope_name] = rnn_state_variables
        if self._target_policy.is_recurrent():
            shapes = self._target_policy.internal_state_shapes()
            rnn_state_variables = create_variables(batch_size, shapes)
            rnn_states[self._target_policy.scope_name] = rnn_state_variables

        training_variables.rnn_states.update(rnn_states)
        return training_variables
