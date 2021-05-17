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

import gym
import numpy as np


@dataclass
class EnvironmentInfo(object):
    """Environment Information class

    This class contains the basic information of the target training environment.
    """
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    max_episode_steps: int

    @staticmethod
    def from_env(env):
        """Create env_info from environment

        Args:
            env (gym.Env): the environment

        Returns:
            EnvironmentInfo\
                (:py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`)

        Example:
            >>> import gym
            >>> from nnabla_rl.environments.environment_info import EnvironmentInfo
            >>> env = gym.make("CartPole-v0")
            >>> env_info = EnvironmentInfo.from_env(env)
            >>> env_info.state_shape
            (4,)
        """
        return EnvironmentInfo(observation_space=env.observation_space,
                               action_space=env.action_space,
                               max_episode_steps=EnvironmentInfo._extract_max_episode_steps(env))

    def is_discrete_action_env(self):
        '''
        Check whether the action to execute in the environment is discrete or not

        Returns:
            bool: True if the action to execute in the environment is discrete. Otherwise False.
        '''
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_action_env(self):
        '''
        Check whether the action to execute in the environment is continuous or not

        Returns:
            bool: True if the action to execute in the environment is continuous. Otherwise False.
        '''
        return not self.is_discrete_action_env()

    @property
    def state_shape(self):
        '''
        The shape of observation space
        '''
        return self.observation_space.shape

    @property
    def state_dim(self):
        '''
        The dimension of state assuming that the state is flatten.
        '''
        return np.prod(self.observation_space.shape)

    @property
    def action_shape(self):
        '''
        The shape of action space
        '''
        return self.action_space.shape

    @property
    def action_dim(self):
        '''
        The dimension of action assuming that the action is flatten.
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
