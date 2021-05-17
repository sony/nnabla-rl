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

import numpy as np
import pytest

import nnabla_rl.environments as E
from nnabla_rl.environments.environment_info import EnvironmentInfo


class TestEnvInfo(object):
    @pytest.mark.parametrize("max_episode_steps", [None, 100, 10000, float('inf')])
    def test_spec_max_episode_steps(self, max_episode_steps):
        dummy_env = E.DummyContinuous(max_episode_steps=max_episode_steps)
        env_info = EnvironmentInfo.from_env(dummy_env)

        if max_episode_steps is None:
            assert env_info.max_episode_steps == float('inf')
        else:
            assert env_info.max_episode_steps == max_episode_steps

    def test_is_discrete_action_env(self):
        dummy_env = E.DummyDiscrete()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.is_discrete_action_env()
        assert not env_info.is_continuous_action_env()

    def test_is_continuous_action_env(self):
        dummy_env = E.DummyContinuous()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert not env_info.is_discrete_action_env()
        assert env_info.is_continuous_action_env()

    def test_state_shape_continuous(self):
        dummy_env = E.DummyContinuous()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.state_shape == dummy_env.observation_space.shape

    def test_state_shape_image(self):
        dummy_env = E.DummyDiscreteImg()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.state_shape == dummy_env.observation_space.shape

    def test_state_dim_continuous(self):
        dummy_env = E.DummyContinuous()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.state_dim == dummy_env.observation_space.shape[0]

    def test_state_dim_image(self):
        dummy_env = E.DummyDiscreteImg()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.state_dim == np.prod(dummy_env.observation_space.shape)

    def test_action_shape_continuous(self):
        dummy_env = E.DummyContinuous()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.action_shape == dummy_env.action_space.shape

    def test_action_shape_discrete(self):
        dummy_env = E.DummyDiscreteImg()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.action_shape == dummy_env.action_space.shape

    def test_action_dim_continuous(self):
        dummy_env = E.DummyContinuous()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.action_dim == dummy_env.action_space.shape[0]

    def test_action_dim_discrete(self):
        dummy_env = E.DummyDiscrete()
        env_info = EnvironmentInfo.from_env(dummy_env)

        assert env_info.action_dim == dummy_env.action_space.n


if __name__ == "__main__":
    pytest.main()
