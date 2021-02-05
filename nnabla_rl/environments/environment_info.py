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

from dataclasses import dataclass

import gym

import numpy as np


@dataclass
class EnvironmentInfo(object):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    max_episode_steps: int

    @staticmethod
    def from_env(env):
        return EnvironmentInfo(observation_space=env.observation_space,
                               action_space=env.action_space,
                               max_episode_steps=EnvironmentInfo._extract_max_episode_steps(env))

    def is_discrete_action_env(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_action_env(self):
        return not self.is_discrete_action_env()

    @property
    def state_shape(self):
        return self.observation_space.shape

    @property
    def state_dim(self):
        '''
        Compute the dimension of state assuming that the state is flatten.
        '''
        return np.prod(self.observation_space.shape)

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def action_dim(self):
        '''
        Compute the dimension of action assuming that the action is flatten.
        '''
        if self.is_discrete_action_env():
            return self.action_space.n
        else:
            return np.prod(self.action_space.shape)

    @staticmethod
    def _extract_max_episode_steps(env):
        if env.spec is None or env.spec.max_episode_steps is None:
            return float("inf")
        else:
            return env.spec.max_episode_steps
