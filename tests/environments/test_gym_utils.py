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

import gym
import numpy as np
import pytest

import nnabla_rl.environments as E
from nnabla_rl.environments.gym_utils import (extract_max_episode_steps, get_space_dim, get_space_shape,
                                              is_same_space_type)


class TestGymUtils():
    def test_get_tuple_space_shape(self):
        tuple_space = gym.spaces.Tuple((gym.spaces.Box(low=0.0, high=1.0, shape=(2, )),
                                        gym.spaces.Box(low=0.0, high=1.0, shape=(3, ))))

        with pytest.raises(ValueError):
            get_space_shape(tuple_space)

    def test_get_box_space_shape(self):
        shape = (5, )
        box_space_shape = gym.spaces.Box(low=0.0, high=1.0, shape=shape)
        actual_shape = get_space_shape(box_space_shape)

        assert actual_shape == shape

    def test_get_discrete_space_shape(self):
        shape = (1, )
        discrete_space_shape = gym.spaces.Discrete(4)
        actual_shape = get_space_shape(discrete_space_shape)

        assert actual_shape == shape

    def test_get_tuple_space_dim(self):
        tuple_space = gym.spaces.Tuple((gym.spaces.Box(low=0.0, high=1.0, shape=(2, )),
                                        gym.spaces.Box(low=0.0, high=1.0, shape=(3, ))))

        with pytest.raises(ValueError):
            get_space_dim(tuple_space)

    def test_get_box_space_dim(self):
        shape = (5, 2)
        box_space_shape = gym.spaces.Box(low=0.0, high=1.0, shape=shape)
        actual_dim = get_space_dim(box_space_shape)

        assert actual_dim == np.prod(shape)

    def test_get_discrete_space_dim(self):
        dim = 4
        discrete_space_shape = gym.spaces.Discrete(dim)
        actual_dim = get_space_shape(discrete_space_shape)

        assert actual_dim == (1, )

    def test_extract_None_max_episode_steps(self):
        env = E.DummyContinuous(max_episode_steps=None)
        actual_max_episode_steps = extract_max_episode_steps(env)

        assert actual_max_episode_steps == float("inf")

    def test_extract_max_episode_steps(self):
        max_episode_steps = 100
        env = E.DummyContinuous(max_episode_steps=max_episode_steps)
        actual_max_episode_steps = extract_max_episode_steps(env)

        assert actual_max_episode_steps == max_episode_steps

    def test_is_same_space_type_box(self):
        box_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, ))
        tuple_box_space = gym.spaces.Tuple((box_space, box_space))

        assert is_same_space_type(tuple_box_space, gym.spaces.Box)

    def test_is_same_space_type_discrete(self):
        discrete_space = gym.spaces.Discrete(4)
        tuple_discrete_space = gym.spaces.Tuple((discrete_space, discrete_space))

        assert is_same_space_type(tuple_discrete_space, gym.spaces.Discrete)

    def test_is_same_space_type_mixed(self):
        box_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, ))
        discrete_space = gym.spaces.Discrete(4)

        tuple_mixed_space = gym.spaces.Tuple((box_space, discrete_space))

        assert not is_same_space_type(tuple_mixed_space, gym.spaces.Box)
        assert not is_same_space_type(tuple_mixed_space, gym.spaces.Discrete)
