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
class BCQQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    lmb: float = 0.75
    num_action_samples: int = 10


class BCQQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]
    _config: BCQQTrainerConfig
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[QFunction, Sequence[QFunction]],
                 target_policy: DeterministicPolicy,
                 env_info: EnvironmentInfo,
                 config: BCQQTrainerConfig = BCQQTrainerConfig()):
        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._assert_no_duplicate_model(self._target_functions)
        self._target_policy = target_policy
        self._prev_target_rnn_states = {}
        self._prev_q_rnn_states = {}
        super(BCQQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        batch_size = training_variables.batch_size
        s_next_rep = RNF.repeat(x=s_next, repeats=self._config.num_action_samples, axis=0)

        train_rnn_states = training_variables.rnn_states
        prev_rnn_states = self._prev_target_rnn_states
        with rnn_support(self._target_policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
            a_next_rep = self._target_policy.pi(s_next_rep)

        q_values = []
        prev_rnn_states = self._prev_q_rnn_states
        for target_q_function in self._target_functions:
            with rnn_support(target_q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_value = target_q_function.q(s_next_rep, a_next_rep)
                q_values.append(q_value)
        q_values = NF.stack(*q_values)

        num_q_ensembles = len(self._target_functions)
        assert isinstance(q_values, nn.Variable)
        assert q_values.shape == (num_q_ensembles, batch_size * self._config.num_action_samples, 1)
        weighted_q_minmax = self._config.lmb * NF.min(q_values, axis=0) + \
            (1.0 - self._config.lmb) * NF.max(q_values, axis=0)
        assert weighted_q_minmax.shape == (batch_size * self._config.num_action_samples, 1)

        next_q_value = NF.max(NF.reshape(weighted_q_minmax, shape=(batch_size, -1)), axis=1, keepdims=True)
        assert next_q_value.shape == (batch_size, 1)
        return reward + gamma * non_terminal * next_q_value

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
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
