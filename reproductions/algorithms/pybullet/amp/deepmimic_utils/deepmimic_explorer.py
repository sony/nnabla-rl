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

from typing import List, Union, cast

import gym

from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import (LinearDecayEpsilonGreedyExplorer,
                                                                     LinearDecayEpsilonGreedyExplorerConfig)
from nnabla_rl.environments.amp_env import AMPEnv, AMPGoalEnv
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.typing import ActionSelector, Experience, State


class ExploreUntilValidEnvironmentExplorer(EnvironmentExplorer):
    def __init__(self, env_info: EnvironmentInfo, config: EnvironmentExplorerConfig = EnvironmentExplorerConfig()):
        super().__init__(env_info, config)

    def step(self, env: gym.Env, n: int = 1, break_if_done: bool = False) -> List[Experience]:
        if break_if_done:
            raise ValueError("DeepMimic Env Explore does not support break_if_done")

        assert 0 < n

        experiences: List[Experience] = []
        if self._state is None:
            self._state = cast(State, env.reset())

        while len(experiences) < n:
            # Instead of using step once, rollout episode and check if the episode is valid.
            tmp_experiences = self.rollout(env)
            _, _, _, non_terminal, _, extra_info = tmp_experiences[-1]

            # the last element is always done
            assert not non_terminal
            # the info of last element should have valid_episode key
            assert "valid_episode" in extra_info

            if extra_info["valid_episode"]:
                experiences.extend(tmp_experiences)
            else:
                # Do not count steps if its not valid episode.
                self._steps -= len(tmp_experiences)

            self._begin_of_episode = not non_terminal

        return experiences


class DeepMimicExplorer(LinearDecayEpsilonGreedyExplorer, ExploreUntilValidEnvironmentExplorer):
    def __init__(self,
                 greedy_action_selector: ActionSelector,
                 random_action_selector: ActionSelector,
                 env_info: EnvironmentInfo,
                 config: LinearDecayEpsilonGreedyExplorerConfig = LinearDecayEpsilonGreedyExplorerConfig()):
        super().__init__(
            env_info=env_info,
            config=config,
            greedy_action_selector=greedy_action_selector,
            random_action_selector=random_action_selector,
        )

    def step(self, env: gym.Env, n: int = 1, break_if_done: bool = False):
        experiences = super().step(env, n, break_if_done)
        assert isinstance(env.unwrapped, AMPEnv) or isinstance(env.unwrapped, AMPGoalEnv)
        casted_env = cast(Union[AMPEnv, AMPGoalEnv], env)
        casted_env.update_sample_counts()
        return experiences
