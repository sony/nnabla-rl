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
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.models import DeterministicPolicy, QFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list


@dataclass
class TD3QTrainerConfig(SquaredTDQFunctionTrainerConfig):
    train_action_noise_sigma: float = 0.2
    train_action_noise_abs: float = 0.5


class TD3QTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]
    _target_policy: DeterministicPolicy
    _config: TD3QTrainerConfig

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[QFunction, Sequence[QFunction]],
                 target_policy: DeterministicPolicy,
                 env_info: EnvironmentInfo,
                 config: TD3QTrainerConfig = TD3QTrainerConfig()):
        self._target_policy = target_policy
        self._target_functions = convert_to_list_if_not_list(target_functions)
        super(TD3QTrainer, self).__init__(train_functions, solvers, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        q_values = []
        a_next = self._compute_noisy_action(s_next)
        a_next.need_grad = False
        for target_q_function in self._target_functions:
            q_value = target_q_function.q(s_next, a_next)
            q_values.append(q_value)
        # Use the minimum among computed q_values by default
        target_q = RF.minimum_n(q_values)
        return reward + gamma * non_terminal * target_q

    def _compute_noisy_action(self, state):
        a_next_var = self._target_policy.pi(state)
        epsilon = NF.clip_by_value(NF.randn(sigma=self._config.train_action_noise_sigma,
                                            shape=a_next_var.shape),
                                   min=-self._config.train_action_noise_abs,
                                   max=self._config.train_action_noise_abs)
        a_tilde_var = a_next_var + epsilon
        return a_tilde_var
