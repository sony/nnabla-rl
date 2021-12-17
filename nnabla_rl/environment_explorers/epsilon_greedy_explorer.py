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
from typing import Dict, Tuple

import numpy as np

from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.typing import ActionSelector


def epsilon_greedy_action_selection(state: np.ndarray,
                                    greedy_action_selector: ActionSelector,
                                    random_action_selector: ActionSelector,
                                    epsilon: float,
                                    *,
                                    begin_of_episode: bool = False):
    if np.random.rand() > epsilon:
        # optimal action
        return greedy_action_selector(state, begin_of_episode=begin_of_episode), True
    else:
        # random action
        return random_action_selector(state, begin_of_episode=begin_of_episode), False


@dataclass
class NoDecayEpsilonGreedyExplorerConfig(EnvironmentExplorerConfig):
    epsilon: float = 1.0

    def __post_init__(self):
        self._assert_between(self.epsilon, 0.0, 1.0, 'epsilon')


class NoDecayEpsilonGreedyExplorer(EnvironmentExplorer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: NoDecayEpsilonGreedyExplorerConfig

    def __init__(self,
                 greedy_action_selector: ActionSelector,
                 random_action_selector: ActionSelector,
                 env_info: EnvironmentInfo,
                 config: NoDecayEpsilonGreedyExplorerConfig = NoDecayEpsilonGreedyExplorerConfig()):
        super().__init__(env_info, config)
        self._greedy_action_selector = greedy_action_selector
        self._random_action_selector = random_action_selector

    def action(self, step: int, state: np.ndarray, *, begin_of_episode: bool = False) -> Tuple[np.ndarray, Dict]:
        epsilon = self._config.epsilon
        (action, info), _ = epsilon_greedy_action_selection(state,
                                                            self._greedy_action_selector,
                                                            self._random_action_selector,
                                                            epsilon,
                                                            begin_of_episode=begin_of_episode)
        return action, info


@dataclass
class LinearDecayEpsilonGreedyExplorerConfig(EnvironmentExplorerConfig):
    """
    List of configurations for Linear decay epsilon-greedy explorer

    Args:
        initial_epsilon (float): Initial value of epsilon. Defaults to 1.0.
        final_epsilon (float): Final value of epsilon after max_explore_steps.
            This value must be smaller than initial_epsilon. Defaults to 0.05.
        max_explore_steps (int): Number of steps to decay epsilon from initial_epsilon to final_epsilon.
            Defaults to 1000000.
    """

    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    max_explore_steps: float = 1000000

    def __post_init__(self):
        self._assert_between(self.initial_epsilon, 0.0, 1.0, 'initial_epsilon')
        self._assert_between(self.final_epsilon, 0.0, 1.0, 'final_epsilon')
        self._assert_descending_order([self.initial_epsilon, self.final_epsilon], 'initial/final epsilon')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')


class LinearDecayEpsilonGreedyExplorer(EnvironmentExplorer):
    '''Linear decay epsilon-greedy explorer

    Epsilon-greedy style explorer. Epsilon is linearly decayed until max_eplore_steps set in the config.

    Args:
        greedy_action_selector (:py:class:`ActionSelector <nnabla_rl.typing.ActionSelector>`):
            callable which computes greedy action with respect to current state.
        random_action_selector (:py:class:`ActionSelector <nnabla_rl.typing.ActionSelector>`):
            callable which computes random action that can be executed in the environment.
        env_info (:py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            environment info
        config (:py:class:`LinearDecayEpsilonGreedyExplorerConfig\
            <nnabla_rl.environment_explorers.LinearDecayEpsilonGreedyExplorerConfig>`): the config of this class.
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: LinearDecayEpsilonGreedyExplorerConfig

    def __init__(self,
                 greedy_action_selector: ActionSelector,
                 random_action_selector: ActionSelector,
                 env_info: EnvironmentInfo,
                 config: LinearDecayEpsilonGreedyExplorerConfig = LinearDecayEpsilonGreedyExplorerConfig()):
        super().__init__(env_info, config)
        self._greedy_action_selector = greedy_action_selector
        self._random_action_selector = random_action_selector

    def action(self, step: int, state: np.ndarray, *, begin_of_episode: bool = False) -> Tuple[np.ndarray, Dict]:
        epsilon = self._compute_epsilon(step)
        (action, info), _ = epsilon_greedy_action_selection(state,
                                                            self._greedy_action_selector,
                                                            self._random_action_selector,
                                                            epsilon,
                                                            begin_of_episode=begin_of_episode)
        return action, info

    def _compute_epsilon(self, step):
        assert 0 <= step
        delta_epsilon = step / self._config.max_explore_steps \
            * (self._config.initial_epsilon - self._config.final_epsilon)
        epsilon = self._config.initial_epsilon - delta_epsilon
        return max(epsilon, self._config.final_epsilon)
