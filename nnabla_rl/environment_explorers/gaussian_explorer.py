# Copyright 2020,2021 Sony Corporation.
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

import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.typing import ActionSelector


@dataclass
class GaussianExplorerConfig(EnvironmentExplorerConfig):
    """
    List of configurations for gaussian explorer.

    Args:
        action_clip_low (float): Minimum noise value. Noise below this value will be clipped.
            Defaults to sys.float_info.min.
        action_clip_high (float): Maximum noise value. Noise above this value will be clipped.
            Defaults to sys.float_info.max.
        sigma (float): Standard deviation of gaussian noise. Must be positive. Defaults to 1.0.
    """

    action_clip_low: float = sys.float_info.min
    action_clip_high: float = sys.float_info.max
    sigma: float = 1.0

    def __post_init__(self):
        self._assert_positive(self.sigma, 'sigma')


class GaussianExplorer(EnvironmentExplorer):
    '''Gaussian explorer

    Explore using policy's action without gaussian noise appended to it. Policy's action must be continuous action.

    Args:
        policy_action_selector (:py:class:`ActionSelector <nnabla_rl.typing.ActionSelector>`):
            callable which computes current policy's action with respect to current state.
        env_info (:py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            environment info
        config (:py:class:`LinearDecayEpsilonGreedyExplorerConfig\
            <nnabla_rl.environment_explorers.LinearDecayEpsilonGreedyExplorerConfig>`): the config of this class.
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: GaussianExplorerConfig

    def __init__(self,
                 policy_action_selector: ActionSelector,
                 env_info: EnvironmentInfo,
                 config: GaussianExplorerConfig = GaussianExplorerConfig()):
        super().__init__(env_info, config)
        self._policy_action_selector = policy_action_selector

    def action(self, step: int, state: np.ndarray, *, begin_of_episode: bool = False) -> Tuple[np.ndarray, Dict]:
        (action, info) = self._policy_action_selector(state, begin_of_episode=begin_of_episode)
        return self._append_noise(action, self._config.action_clip_low, self._config.action_clip_high), info

    def _append_noise(self, action, low, high):
        noise = np.random.normal(loc=0.0, scale=self._config.sigma, size=action.shape).astype(np.float32)
        return np.clip(action + noise, low, high)
