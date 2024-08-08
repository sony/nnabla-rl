# Copyright 2024 Sony Group Corporation.
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

from typing import List, Tuple

import numpy as np

from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.preprocessors.running_mean_normalizer import RunningMeanNormalizer


class DeepMimicTupleRunningMeanNormalizer(Preprocessor, Model):
    _normalizers: List[RunningMeanNormalizer]

    def __init__(
        self,
        scope_name: str,
        policy_state_shape: Tuple[Tuple[int, ...], ...],
        reward_state_shape: Tuple[Tuple[int, ...], ...],
        policy_state_mean_initializer: np.ndarray,
        policy_state_var_initializer: np.ndarray,
        reward_state_mean_initializer: np.ndarray,
        reward_state_var_initializer: np.ndarray,
        epsilon: float = 1e-2,
        mode_for_floating_point_error: str = "max",
    ):
        super(DeepMimicTupleRunningMeanNormalizer, self).__init__(scope_name)

        self._normalizers = []
        policy_state_normalizer = RunningMeanNormalizer(
            scope_name + "/policy_state",
            shape=policy_state_shape,
            epsilon=epsilon,
            mode_for_floating_point_error=mode_for_floating_point_error,
            mean_initializer=policy_state_mean_initializer,
            var_initializer=policy_state_var_initializer,
        )
        reward_state_normalizer = RunningMeanNormalizer(
            scope_name + "/reward_state",
            shape=reward_state_shape,
            epsilon=epsilon,
            mode_for_floating_point_error=mode_for_floating_point_error,
            mean_initializer=reward_state_mean_initializer,
            var_initializer=reward_state_var_initializer,
        )
        self._normalizers = [policy_state_normalizer, reward_state_normalizer]

    def process(self, x):
        assert len(x) == 3
        normalized_policy_state = self._normalizers[0].process(x[0])
        normalized_reward_state = self._normalizers[1].process(x[1])
        return (normalized_policy_state, normalized_reward_state, x[2])

    def update(self, data):
        assert len(data) == 3
        self._normalizers[0].update(np.array(data[0], dtype=np.float32))
        # NOTE: Only use valid reward state
        mask = (data[2] == 1.0).flatten()
        self._normalizers[1].update(np.array(data[1][mask], dtype=np.float32))


class DeepMimicGoalTupleRunningMeanNormalizer(DeepMimicTupleRunningMeanNormalizer):
    _normalizers: List[RunningMeanNormalizer]

    def __init__(
        self,
        scope_name: str,
        policy_state_shape: Tuple[Tuple[int, ...], ...],
        reward_state_shape: Tuple[Tuple[int, ...], ...],
        goal_state_shape: Tuple[Tuple[int, ...], ...],
        policy_state_mean_initializer: np.ndarray,
        policy_state_var_initializer: np.ndarray,
        reward_state_mean_initializer: np.ndarray,
        reward_state_var_initializer: np.ndarray,
        goal_state_mean_initializer: np.ndarray,
        goal_state_var_initializer: np.ndarray,
        epsilon: float = 1e-2,
        mode_for_floating_point_error: str = "max",
    ):
        super().__init__(
            scope_name,
            policy_state_shape,
            reward_state_shape,
            policy_state_mean_initializer,
            policy_state_var_initializer,
            reward_state_mean_initializer,
            reward_state_var_initializer,
            epsilon,
            mode_for_floating_point_error,
        )
        goal_state_normalizer = RunningMeanNormalizer(
            scope_name + "/goal_state",
            shape=goal_state_shape,
            epsilon=epsilon,
            mode_for_floating_point_error=mode_for_floating_point_error,
            mean_initializer=goal_state_mean_initializer,
            var_initializer=goal_state_var_initializer,
        )
        self._normalizers.append(goal_state_normalizer)

    def process(self, x):
        assert len(x) == 7
        normalized_policy_state = self._normalizers[0].process(x[0])
        normalized_reward_state = self._normalizers[1].process(x[1])
        normalized_goal_state = self._normalizers[2].process(x[3])
        return (normalized_policy_state, normalized_reward_state, x[2], normalized_goal_state, x[4], x[5], x[6])

    def update(self, data):
        assert len(data) == 7
        self._normalizers[0].update(np.array(data[0], dtype=np.float32))
        # NOTE: Only use valid reward state
        mask = (data[2] == 1.0).flatten()
        self._normalizers[1].update(np.array(data[1][mask], dtype=np.float32))

        # NOTE: In deepmimic env, expert goals are dummy, so skip to use it.
        if np.all(data[4] == 1.0):
            self._normalizers[2].update(np.array(data[3], dtype=np.float32))
