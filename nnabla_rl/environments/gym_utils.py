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

from typing import Tuple, Union

import gym
import numpy as np


def get_space_shape(space: gym.spaces.Space) -> Tuple[int, ...]:
    if isinstance(space, gym.spaces.Box):
        return tuple(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return (1, )
    else:
        raise ValueError


def get_space_dim(space: gym.spaces.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    else:
        raise ValueError


def get_space_high(space: gym.spaces.Space) -> Union[np.ndarray, str]:
    if isinstance(space, gym.spaces.Box):
        return space.high
    elif isinstance(space, gym.spaces.Discrete):
        return 'N/A'
    else:
        raise ValueError


def get_space_low(space: gym.spaces.Space) -> Union[np.ndarray, str]:
    if isinstance(space, gym.spaces.Box):
        return space.low
    elif isinstance(space, gym.spaces.Discrete):
        return 'N/A'
    else:
        raise ValueError


def extract_max_episode_steps(env_or_env_info):
    if isinstance(env_or_env_info, gym.Env):
        if env_or_env_info.spec is None or env_or_env_info.spec.max_episode_steps is None:
            return float("inf")
        else:
            return env_or_env_info.spec.max_episode_steps
    else:
        return env_or_env_info.max_episode_steps


def is_same_space_type(query_space: gym.spaces.Space,
                       key_space: Union[gym.spaces.Discrete, gym.spaces.Box]) -> bool:
    '''
    Check whether the query_space has the same type of key_space or not.
    Note that if the query_space is gym.spaces.Tuple, this method checks
    whether all of the element of the query_space are the key_space or not.

    Args:
        query_space (gym.spaces.Space): space
        key_space (Union[gym.spaces.Discrete, gym.spaces.Box]): space
    Returns:
        bool: True if the query_space is the same as key_space. Otherwise False.
    '''
    if isinstance(query_space, gym.spaces.Tuple):
        return all(isinstance(s, key_space) for s in query_space)
    else:
        return isinstance(query_space, key_space)
