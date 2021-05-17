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

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo


def epsilon_greedy_action_selection(state, greedy_action_selector, random_action_selector, epsilon):
    if np.random.rand() > epsilon:
        # optimal action
        return greedy_action_selector(state), True
    else:
        # random action
        return random_action_selector(state), False


@dataclass
class LinearDecayEpsilonGreedyExplorerConfig(EnvironmentExplorerConfig):
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    max_explore_steps: float = 1000000

    def __post_init__(self):
        self._assert_between(self.initial_epsilon, 0.0, 1.0, 'initial_epsilon')
        self._assert_between(self.final_epsilon, 0.0, 1.0, 'final_epsilon')
        self._assert_descending_order([self.initial_epsilon, self.final_epsilon], 'initial/final epsilon')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')


class LinearDecayEpsilonGreedyExplorer(EnvironmentExplorer):
    def __init__(self,
                 greedy_action_selector: Callable[[np.array], Tuple[np.array, Dict]],
                 random_action_selector: Callable[[np.array], Tuple[np.array, Dict]],
                 env_info: EnvironmentInfo,
                 config: LinearDecayEpsilonGreedyExplorerConfig = LinearDecayEpsilonGreedyExplorerConfig()):
        super().__init__(env_info, config)
        self._greedy_action_selector = greedy_action_selector
        self._random_action_selector = random_action_selector

    def action(self, step, state):
        epsilon = self._compute_epsilon(step)
        (action, info), _ = epsilon_greedy_action_selection(state,
                                                            self._greedy_action_selector,
                                                            self._random_action_selector,
                                                            epsilon)
        return action, info

    def _compute_epsilon(self, step):
        assert 0 <= step
        delta_epsilon = step / self._config.max_explore_steps \
            * (self._config.initial_epsilon - self._config.final_epsilon)
        epsilon = self._config.initial_epsilon - delta_epsilon
        return max(epsilon, self._config.final_epsilon)
