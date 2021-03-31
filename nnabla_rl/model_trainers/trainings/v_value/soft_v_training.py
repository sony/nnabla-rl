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
from nnabla_rl.models import Model, QFunction, StochasticPolicy, VFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list


class _VFunctionSoftVTraining(Training):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]
    _target_policy: StochasticPolicy

    def __init__(self,
                 target_functions: Sequence[QFunction],
                 target_policy: StochasticPolicy):
        self._target_functions = target_functions
        self._target_policy = target_policy

    def compute_target(self, training_variables: TrainingVariables, **kwargs):
        s_current = training_variables.s_current

        policy_distribution = self._target_policy.pi(s_current)
        sampled_action, log_pi = policy_distribution.sample_and_compute_log_prob()

        q_values = []
        for q_function in self._target_functions:
            q_values.append(q_function.q(s_current, sampled_action))
        target_q = RNF.minimum_n(q_values)

        return target_q - log_pi


class SoftVTraining(Training):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _delegate: Training

    def __init__(self,
                 train_functions: Union[Sequence[Model], Model],
                 target_functions: Union[Sequence[Model], Model],
                 target_policy: StochasticPolicy):
        super(SoftVTraining, self).__init__()
        train_functions = convert_to_list_if_not_list(train_functions)
        target_functions = convert_to_list_if_not_list(target_functions)
        train_function = train_functions[0]
        target_function = target_functions[0]
        if isinstance(train_function, VFunction) and isinstance(target_function, QFunction):
            target_functions = cast(Sequence[QFunction], target_functions)
            self._delegate = _VFunctionSoftVTraining(target_functions, target_policy)
        else:
            raise NotImplementedError(f'No training implementation for class: {train_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
