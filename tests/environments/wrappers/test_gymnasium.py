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

import numpy as np
import pytest

from nnabla_rl.environments.dummy import DummyContinuous, DummyGymnasiumMujocoEnv
from nnabla_rl.environments.wrappers.common import NumpyFloat32Env
from nnabla_rl.environments.wrappers.gymnasium import Gymnasium2GymWrapper

max_episode_steps = 10


class TestGymnasium(object):
    def test_gym_env(self):
        env = DummyContinuous(max_episode_steps=max_episode_steps)
        with pytest.raises(ValueError):
            env = Gymnasium2GymWrapper(env)

    def test_gym_wrapper(self):
        env = DummyContinuous(max_episode_steps=max_episode_steps)
        env = NumpyFloat32Env(env)
        with pytest.raises(ValueError):
            env = Gymnasium2GymWrapper(env)

    def test_reset(self):
        env = DummyGymnasiumMujocoEnv(max_episode_steps=max_episode_steps)
        raw_reset_outputs = env.reset()
        assert isinstance(raw_reset_outputs, tuple)
        assert isinstance(raw_reset_outputs[0], np.ndarray)
        assert isinstance(raw_reset_outputs[1], dict)

        wrapped_env = Gymnasium2GymWrapper(env)
        reset_outputs = wrapped_env.reset()
        assert isinstance(reset_outputs, np.ndarray)

    def test_step(self):
        env = DummyGymnasiumMujocoEnv(max_episode_steps=max_episode_steps)
        action = env.action_space.sample()
        raw_step_outputs = env.step(action)
        assert isinstance(raw_step_outputs, tuple)
        assert len(raw_step_outputs) == 5

        wrapped_env = Gymnasium2GymWrapper(env)
        action = wrapped_env.action_space.sample()
        step_outputs = wrapped_env.step(action)
        assert isinstance(step_outputs, tuple)
        assert len(step_outputs) == 4


if __name__ == "__main__":
    pytest.main()
