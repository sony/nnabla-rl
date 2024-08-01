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

from typing import Optional

import gym
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces
from gymnasium.utils import seeding


class Gymnasium2GymWrapper(gym.Wrapper):
    def __init__(self, env):
        if isinstance(env, gym.Env) or isinstance(env, gym.Wrapper):
            raise ValueError("'env' should not be an instance of 'gym.Env' and 'gym.Wrapper'")

        super().__init__(env)

        # observation space
        if isinstance(env.observation_space, gymnasium_spaces.Tuple):
            self.observation_space = gym_spaces.Tuple(
                [self._translate_space(observation_space) for observation_space in env.observation_space]
            )
        elif isinstance(env.observation_space, gymnasium_spaces.Dict):
            self.observation_space = gym_spaces.Dict(
                {
                    key: self._translate_space(observation_space)
                    for key, observation_space in env.observation_space.items()
                }
            )
        else:
            self.observation_space = self._translate_space(env.observation_space)

        # action space
        if isinstance(env.action_space, gymnasium_spaces.Tuple):
            self.action_space = gym_spaces.Tuple(
                [self._translate_space(action_space) for action_space in env.action_space]
            )
        elif isinstance(env.action_space, gymnasium_spaces.Dict):
            self.action_space = gym_spaces.Dict(
                {key: self._translate_space(action_space) for key, action_space in env.action_space.items()}
            )
        else:
            self.action_space = self._translate_space(env.action_space)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info.update({"TimeLimit.truncated": truncated})
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None):
        np_random, seed = seeding.np_random(seed)
        self.env.np_random = np_random  # type: ignore
        return [seed]

    @property
    def unwrapped(self):
        return self

    def _translate_space(self, space):
        if isinstance(space, gymnasium_spaces.Box):
            return gym_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gymnasium_spaces.Discrete):
            return gym_spaces.Discrete(n=int(space.n))
        else:
            raise NotImplementedError
