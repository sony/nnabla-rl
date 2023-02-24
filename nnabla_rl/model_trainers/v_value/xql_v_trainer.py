# Copyright 2023 Sony Group Corporation.
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
from typing import Dict, Optional, Sequence, Union

import numpy as np

import nnabla as nn
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch, TrainingVariables, rnn_support
from nnabla_rl.model_trainers.v_value.extreme_v_function_trainer import (ExtremeVFunctionTrainer,
                                                                         ExtremeVFunctionTrainerConfig)
from nnabla_rl.models import Model, QFunction, StochasticPolicy, VFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class XQLVTrainerConfig(ExtremeVFunctionTrainerConfig):
    pass


class XQLVTrainer(ExtremeVFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ExtremeVFunctionTrainerConfig
    _prev_q_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_pi_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[QFunction, Sequence[QFunction]],
                 env_info: EnvironmentInfo,
                 target_policy: Optional[StochasticPolicy] = None,
                 config: XQLVTrainerConfig = XQLVTrainerConfig()):
        self._target_policy = target_policy
        self._prev_pi_rnn_states = {}

        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._prev_q_rnn_states = {}

        super(XQLVTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.a_current, b.a_current)

        return super()._update_model(models, solvers, batch, training_variables, **kwargs)

    def _compute_target(self, training_variables: TrainingVariables):
        s_current = training_variables.s_current
        if self._target_policy is None:
            a_current = training_variables.a_current
        else:
            train_rnn_states = training_variables.rnn_states
            prev_rnn_states = self._prev_pi_rnn_states
            with rnn_support(self._target_policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
                a_current = self._target_policy.pi(s_current).sample()

        q_values = []
        train_rnn_states = training_variables.rnn_states
        prev_rnn_states = self._prev_q_rnn_states
        for q_function in self._target_functions:
            with rnn_support(q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_values.append(q_function.q(s_current, a_current))
        return RNF.minimum_n(q_values)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)
        training_variables.a_current = create_variable(batch_size, self._env_info.action_shape)

        return training_variables
