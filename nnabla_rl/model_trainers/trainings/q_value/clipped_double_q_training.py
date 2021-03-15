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

from typing import Sequence, Union, cast

import nnabla as nn
import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import Model, QFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list


class _QFunctionClippedDoubleQTraining(Training):
    _target_functions: Sequence[QFunction]

    def __init__(self, target_functions: Sequence[QFunction]):
        self._target_functions = target_functions

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
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


class ClippedDoubleQTraining(Training):
    _delegate: Training

    def __init__(self,
                 train_functions: Union[Model, Sequence[Model]],
                 target_functions: Union[Model, Sequence[Model]]):
        train_functions = convert_to_list_if_not_list(train_functions)
        target_functions = convert_to_list_if_not_list(target_functions)
        if len(target_functions) < 2:
            raise ValueError('Must have at least 2 target functions for training')
        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, QFunction) and isinstance(target_function, QFunction):
            target_functions = cast(Sequence[QFunction], target_functions)
            self._delegate = _QFunctionClippedDoubleQTraining(target_functions)
        else:
            raise NotImplementedError(f'No training implementation for class: {target_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
