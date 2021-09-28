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
from typing import Any, Callable, Dict, Optional

import gym

from nnabla_rl.environments.gym_utils import (extract_max_episode_steps, get_space_dim, get_space_high, get_space_low,
                                              get_space_shape, is_same_space_type)


@dataclass
class EnvironmentInfo(object):
    """Environment Information class

    This class contains the basic information of the target training environment.
    """
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    max_episode_steps: int

    def __init__(self,
                 observation_space,
                 action_space,
                 max_episode_steps,
                 unwrapped_env,
                 reward_function: Optional[Callable[[Any, Any, Dict], int]] = None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps
        self.unwrapped_env = unwrapped_env
        self.reward_function = reward_function

        if not (self.is_discrete_state_env() or self.is_continuous_state_env()):
            raise ValueError("Unsupported state space")

        if not (self.is_discrete_action_env() or self.is_continuous_action_env()):
            raise ValueError("Unsupported action space")

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
        reward_function = env.compute_reward if hasattr(env, 'compute_reward') else None
        unwrapped_env = env.unwrapped
        return EnvironmentInfo(observation_space=env.observation_space,
                               action_space=env.action_space,
                               max_episode_steps=extract_max_episode_steps(env),
                               unwrapped_env=unwrapped_env,
                               reward_function=reward_function)

    def is_discrete_action_env(self):
        '''
        Check whether the action to execute in the environment is discrete or not

        Returns:
            bool: True if the action to execute in the environment is discrete. Otherwise False.
                Note that if the action is gym.spaces.Tuple and all of the element are discrete, it returns True.
        '''
        return is_same_space_type(self.action_space, gym.spaces.Discrete)

    def is_continuous_action_env(self):
        '''
        Check whether the action to execute in the environment is continuous or not

        Returns:
            bool: True if the action to execute in the environment is continuous. Otherwise False.
                Note that if the action is gym.spaces.Tuple and all of the element are continuous, it returns True.
        '''
        return is_same_space_type(self.action_space, gym.spaces.Box)

    def is_discrete_state_env(self):
        '''
        Check whether the state of the environment is discrete or not

        Returns:
            bool: True if the state of the environment is discrete. Otherwise False.
                Note that if the state is gym.spaces.Tuple and all of the element are discrete, it returns True.
        '''
        return is_same_space_type(self.observation_space, gym.spaces.Discrete)

    def is_continuous_state_env(self):
        '''
        Check whether the state of the environment is continuous or not

        Returns:
            bool: True if the state of the environment is continuous. Otherwise False.
                Note that if the state is gym.spaces.Tuple and all of the element are continuous, it returns True.
        '''
        return is_same_space_type(self.observation_space, gym.spaces.Box)

    def is_tuple_state_env(self):
        '''
        Check whether the state of the environment is tuple or not

        Returns:
            bool: True if the state of the environment is tuple. Otherwise False.
        '''
        return isinstance(self.observation_space, gym.spaces.Tuple)

    def is_goal_conditioned_env(self):
        '''
        Check whether the environment is gym.GoalEnv or not

        Returns:
            bool: True if the environment is gym.GoalEnv. Otherwise False.
        '''
        return issubclass(self.unwrapped_env.__class__, gym.GoalEnv)

    @property
    def state_shape(self):
        '''
        The shape of observation space
        '''
        if self.is_tuple_state_env():
            return tuple(map(get_space_shape, self.observation_space))
        else:
            return get_space_shape(self.observation_space)

    @property
    def state_dim(self):
        '''
        The dimension of state assuming that the state is flatten.
        '''
        if self.is_tuple_state_env():
            return tuple(map(get_space_dim, self.observation_space))
        else:
            return get_space_dim(self.observation_space)

    @property
    def state_high(self):
        '''
        The upper limit of observation space
        '''
        if self.is_tuple_state_env():
            return tuple(map(get_space_high, self.observation_space))
        else:
            return get_space_high(self.observation_space)

    @property
    def state_low(self):
        '''
        The lower limit of observation space
        '''
        if self.is_tuple_state_env():
            return tuple(map(get_space_low, self.observation_space))
        else:
            return get_space_low(self.observation_space)

    @property
    def action_high(self):
        '''
        The upper limit of action space
        '''
        return get_space_high(self.action_space)

    @property
    def action_low(self):
        '''
        The lower limit of action space
        '''
        return get_space_low(self.action_space)

    @property
    def action_shape(self):
        '''
        The shape of action space
        '''
        return get_space_shape(self.action_space)

    @property
    def action_dim(self):
        '''
        The dimension of action assuming that the action is flatten.
        '''
        return get_space_dim(self.action_space)
