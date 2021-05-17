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
from nnabla_rl.environments.wrappers.common import NumpyFloat32Env


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
        assert env.observation_space.dtype == np.float32
        assert not env.action_space.dtype == np.float32

        action = env.action_space.sample()
        next_state, reward, _, _ = env.step(action)

        assert next_state.dtype == np.float32
        assert reward.dtype == np.float32


if __name__ == "__main__":
    pytest.main()
