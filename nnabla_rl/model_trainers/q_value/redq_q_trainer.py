# Copyright 2022,2023,2024 Sony Group Corporation.
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
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (
    SquaredTDQFunctionTrainer,
    SquaredTDQFunctionTrainerConfig,
)
from nnabla_rl.models import QFunction, StochasticPolicy
from nnabla_rl.utils.data import convert_to_list_if_not_list
from nnabla_rl.utils.misc import create_variables


@dataclass
class REDQQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    M: int = 2


class REDQQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]
    _target_policy: StochasticPolicy
    _config: REDQQTrainerConfig
    _temperature: nn.Variable
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(
        self,
        train_functions: Union[QFunction, Sequence[QFunction]],
        solvers: Dict[str, nn.solver.Solver],
        target_functions: Union[QFunction, Sequence[QFunction]],
        target_policy: StochasticPolicy,
        temperature: nn.Variable,
        env_info: EnvironmentInfo,
        config: REDQQTrainerConfig = REDQQTrainerConfig(),
    ):
        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._N = len(self._target_functions)
        self._assert_no_duplicate_model(self._target_functions)
        self._target_policy = target_policy
        self._temperature = temperature
        self._prev_target_rnn_states = {}
        self._prev_q_rnn_states = {}
        super(REDQQTrainer, self).__init__(train_functions, solvers, env_info, config)

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
            policy_distribution = self._target_policy.pi(s_next)
        a_next, log_pi = policy_distribution.sample_and_compute_log_prob()

        q_values = []
        prev_rnn_states = self._prev_q_rnn_states
        for target_q_function in self._target_functions:
            with rnn_support(target_q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_value = target_q_function.q(s_next, a_next)
                q_values.append(q_value)

        q_values = NF.stack(*q_values, axis=len(q_values[0].shape) - 1)
        indices = NF.randint(low=0, high=self._N, shape=(1, self._config.M))
        indices = NF.broadcast(indices, shape=(training_variables.batch_size, self._config.M))

        q_values = NF.gather(q_values, indices, axis=1, batch_dims=1)
        target_q = NF.min(q_values, axis=1)
        return reward + gamma * non_terminal * (target_q - self._temperature * log_pi)

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
