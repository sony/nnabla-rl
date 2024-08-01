# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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
from nnabla_rl.environments.wrappers.common import FlattenNestedTupleStateWrapper, NumpyFloat32Env, TimestepAsStateEnv


class DummyNestedTupleStateEnv(gym.Env):
    def __init__(self, observation_space) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))
        self.observation_space = observation_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, a):
        next_state = self.observation_space.sample()
        reward = np.random.randn()
        done = False
        info = {}
        return next_state, reward, done, info


class TestCommon(object):
    def test_numpy_float32_env_continuous(self):
        env = E.DummyContinuous()
        env = NumpyFloat32Env(env)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.dtype == np.float32

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state.dtype == np.float32
        assert reward.dtype == np.float32

    def test_numpy_float32_env_discrete(self):
        env = E.DummyDiscrete()
        env = NumpyFloat32Env(env)
        assert not env.action_space.dtype == np.float32

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state.dtype == np.float32
        assert reward.dtype == np.float32

    def test_numpy_float32_env_tuple_continuous(self):
        env = E.DummyTupleContinuous()
        env = NumpyFloat32Env(env)

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state[0].dtype == np.float32
        assert next_state[1].dtype == np.float32
        assert action[0].dtype == np.float32
        assert action[1].dtype == np.float32
        assert reward.dtype == np.float32

    def test_numpy_float32_env_tuple_discrete(self):
        env = E.DummyTupleDiscrete()
        env = NumpyFloat32Env(env)

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state[0].dtype == np.float32
        assert next_state[1].dtype == np.float32
        assert isinstance(action[0], int)
        assert isinstance(action[0], int)
        assert reward.dtype == np.float32

    def test_numpy_float32_env_tuple_mixed(self):
        env = E.DummyTupleMixed()
        env = NumpyFloat32Env(env)

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state[0].dtype == np.float32
        assert next_state[1].dtype == np.float32
        assert isinstance(action[0], int)
        assert action[1].dtype == np.float32
        assert reward.dtype == np.float32

    def test_timestep_as_state_env_continuous(self):
        env = E.DummyContinuous()
        env = TimestepAsStateEnv(env)

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert len(next_state) == 2
        assert next_state[1] == 1

        next_state, reward, _, _ = env.step(action)

        assert len(next_state) == 2
        assert next_state[1] == 2

    def test_timestep_as_state_env_discrete(self):
        env = E.DummyDiscrete()
        env = TimestepAsStateEnv(env)

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert len(next_state) == 2
        assert next_state[1] == 1

        next_state, reward, _, _ = env.step(action)

        assert len(next_state) == 2
        assert next_state[1] == 2

    def test_flatten_nested_tuple_state(self):
        box_space_list = [gym.spaces.Box(low=0.0, high=1.0, shape=(i,)) for i in range(5)]

        observation_space = gym.spaces.Tuple(
            [gym.spaces.Tuple(box_space_list[0:3]), gym.spaces.Tuple(box_space_list[3:])]
        )

        env = DummyNestedTupleStateEnv(observation_space)
        env = FlattenNestedTupleStateWrapper(env)

        # Check observation space is flattened
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        for i, actual_space in enumerate(env.observation_space):
            assert isinstance(actual_space, gym.spaces.Box)
            assert actual_space.shape == box_space_list[i].shape

        # Check state shape
        state = env.reset()
        self._nested_shape_check(state, box_space_list)

        next_state, _, _, _ = env.step(np.array([1.0]))
        self._nested_shape_check(state, box_space_list)

    def _nested_shape_check(self, state, box_space_list):
        assert isinstance(state, tuple)
        for s, space in zip(state, box_space_list):
            assert s.shape == space.shape


if __name__ == "__main__":
    pytest.main()
