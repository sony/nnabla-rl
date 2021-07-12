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
from typing import Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.q_value.categorical_dqn_q_trainer import (CategoricalDQNQTrainer,
                                                                        CategoricalDQNQTrainerConfig)
from nnabla_rl.models import ValueDistributionFunction


@dataclass
class CategoricalDDQNQTrainerConfig(CategoricalDQNQTrainerConfig):
    pass


class CategoricalDDQNQTrainer(CategoricalDQNQTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: ValueDistributionFunction

    def __init__(self,
                 train_function: ValueDistributionFunction,
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: ValueDistributionFunction,
                 env_info: EnvironmentInfo,
                 config: CategoricalDDQNQTrainerConfig = CategoricalDDQNQTrainerConfig()):
        self._train_function = train_function
        self._target_function = target_function
        super(CategoricalDDQNQTrainer, self).__init__(train_function, solvers, target_function, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        N = self._target_function._n_atom
        v_max = self._config.v_max
        v_min = self._config.v_min

        a_next = self._train_function.as_q_function().argmax_q(s_next)
        pj = self._target_function.probs(s_next, a_next)

        delta_z = (v_max - v_min) / (N - 1)
        z = np.asarray([v_min + i * delta_z for i in range(N)])
        z = np.broadcast_to(array=z, shape=(batch_size, N))
        z = nn.Variable.from_numpy_array(z)
        target = reward + non_terminal * gamma * z
        Tz = NF.clip_by_value(target, v_min, v_max)
        assert Tz.shape == (batch_size, N)

        mi = self._compute_projection(Tz, pj, N, v_max, v_min)
        return mi
