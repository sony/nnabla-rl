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

from typing import cast, Union, Sequence

import nnabla as nn
import nnabla.functions as NF

import nnabla_rl.functions as RF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import QFunction, DeterministicPolicy, Model
from nnabla_rl.utils.data import convert_to_list_if_not_list


class _QFunctionTD3Training(Training):
    _target_functions: Sequence[QFunction]
    _target_policy: DeterministicPolicy
    _train_action_noise_sigma: float
    _train_action_noise_abs: float

    def __init__(self,
                 target_functions: Sequence[QFunction],
                 target_policy: DeterministicPolicy,
                 train_action_noise_sigma: float = 0.2,
                 train_action_noise_abs: float = 0.5):
        self._target_functions = target_functions
        self._target_policy = target_policy
        self._train_action_noise_sigma = train_action_noise_sigma
        self._train_action_noise_abs = train_action_noise_abs

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
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
        epsilon = NF.clip_by_value(RF.randn(sigma=self._train_action_noise_sigma,
                                            shape=a_next_var.shape),
                                   min=-self._train_action_noise_abs,
                                   max=self._train_action_noise_abs)
        a_tilde_var = a_next_var + epsilon
        return a_tilde_var


class TD3Training(Training):
    def __init__(self,
                 train_functions: Union[Model, Sequence[Model]],
                 target_functions: Union[Model, Sequence[Model]],
                 target_policy: DeterministicPolicy,
                 train_action_noise_sigma: float = 0.2,
                 train_action_noise_abs: float = 0.5):
        train_functions = convert_to_list_if_not_list(train_functions)
        target_functions = convert_to_list_if_not_list(target_functions)
        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, QFunction) and isinstance(target_function, QFunction):
            target_functions = cast(Sequence[QFunction], target_functions)
            self._delegate = _QFunctionTD3Training(target_functions,
                                                   target_policy,
                                                   train_action_noise_sigma,
                                                   train_action_noise_abs)
        else:
            raise NotImplementedError(f'No training implementation for class: {target_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
