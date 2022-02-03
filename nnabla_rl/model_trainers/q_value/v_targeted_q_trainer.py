# Copyright 2021,2022 Sony Group Corporation.
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
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import QFunction, VFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list
from nnabla_rl.utils.misc import create_variables


@dataclass
class VTargetedQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    """
    List of VTargetedQTrainer configuration.

    Args:
        pure_exploration (bool): If True, compute q-value target without adding reward. \
            :math:`target=\\gamma\\times V(s_{t+1})`.\
            Used in Disentangled MME.\
    """
    pure_exploration: bool = False


class VTargetedQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[VFunction]
    _target_v_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _config: VTargetedQTrainerConfig

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[VFunction, Sequence[VFunction]],
                 env_info: EnvironmentInfo,
                 config: VTargetedQTrainerConfig = VTargetedQTrainerConfig()):
        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._assert_no_duplicate_model(self._target_functions)
        self._target_v_rnn_states = {}
        super(VTargetedQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        target_vs = []
        prev_rnn_states = self._target_v_rnn_states
        train_rnn_states = training_variables.rnn_states
        for v_function in self._target_functions:
            with rnn_support(v_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                target_vs.append(v_function.v(s_next))
        target_v = RNF.minimum_n(target_vs)
        if self._config.pure_exploration:
            return gamma * non_terminal * target_v
        else:
            return reward + gamma * non_terminal * target_v

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        rnn_states = {}
        for target_function in self._target_functions:
            if target_function.is_recurrent():
                shapes = target_function.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[target_function.scope_name] = rnn_state_variables
        training_variables.rnn_states.update(rnn_states)
        return training_variables
