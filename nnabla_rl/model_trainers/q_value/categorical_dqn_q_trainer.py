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

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.value_distribution_function_trainer import (
    ValueDistributionFunctionTrainer, ValueDistributionFunctionTrainerConfig)
from nnabla_rl.models import ValueDistributionFunction
from nnabla_rl.utils.misc import create_variables


@dataclass
class CategoricalDQNQTrainerConfig(ValueDistributionFunctionTrainerConfig):
    pass


class CategoricalDQNQTrainer(ValueDistributionFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: ValueDistributionFunction
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[ValueDistributionFunction, Sequence[ValueDistributionFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: ValueDistributionFunction,
                 env_info: EnvironmentInfo,
                 config: CategoricalDQNQTrainerConfig = CategoricalDQNQTrainerConfig()):
        self._target_function = target_function
        self._prev_target_rnn_states = {}
        super(CategoricalDQNQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        N = self._target_function._n_atom
        v_max = self._config.v_max
        v_min = self._config.v_min

        prev_rnn_states = self._prev_target_rnn_states
        train_rnn_states = training_variables.rnn_states
        with rnn_support(self._target_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
            pj = self._target_function.max_q_probs(s_next)

        delta_z = (v_max - v_min) / (N - 1)
        z = np.asarray([v_min + i * delta_z for i in range(N)])
        z = np.broadcast_to(array=z, shape=(batch_size, N))
        z = nn.Variable.from_numpy_array(z)
        target = reward + non_terminal * gamma * z
        Tz = NF.clip_by_value(target, v_min, v_max)
        assert Tz.shape == (batch_size, N)

        mi = self._compute_projection(Tz, pj, N, v_max, v_min)
        return mi

    def _compute_projection(self, Tz, pj, N, v_max, v_min):
        batch_size = Tz.shape[0]
        delta_z = (v_max - v_min) / (N - 1)

        bj = (Tz - v_min) / delta_z
        bj = NF.clip_by_value(bj, 0, N - 1)

        lower = NF.floor(bj)
        upper = NF.ceil(bj)

        ml_indices = lower
        mu_indices = upper

        mi = nn.Variable.from_numpy_array(np.zeros(shape=(batch_size, N), dtype=np.float32))
        # Fix upper - bj = bj - lower = 0 (Prevent not getting both 0. upper - l must always be 1)
        # upper - bj = (1 + lower) - bj
        upper = 1 + lower

        result_upper = NF.scatter_add(mi, ml_indices, pj * (upper - bj), axis=-1)
        result_lower = NF.scatter_add(mi, mu_indices, pj * (bj - lower), axis=-1)

        return (result_upper + result_lower)

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        rnn_states = {}
        if self._target_function.is_recurrent():
            shapes = self._target_function.internal_state_shapes()
            rnn_state_variables = create_variables(batch_size, shapes)
            rnn_states[self._target_function.scope_name] = rnn_state_variables

        training_variables.rnn_states.update(rnn_states)
        return training_variables
