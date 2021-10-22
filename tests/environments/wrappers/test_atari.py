
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

from nnabla_rl.environments.dummy import DummyAtariEnv
from nnabla_rl.environments.wrappers.atari import FlickerFrame


class TestFlickerFrame(object):
    def test_always_obscure(self):
        env = DummyAtariEnv()
        env = FlickerFrame(env, flicker_probability=1.0)

        # frame will be always obscured
        state = env.reset()
        np.testing.assert_allclose(state, 0.0)
        for _ in range(10):
            action = env.action_space.sample()
            next_state, *_ = env.step(action)
            np.testing.assert_allclose(next_state, 0.0)

    def test_never_obscure(self):
        env = DummyAtariEnv()
        env = FlickerFrame(env, flicker_probability=0.0)

        # frame will be always obscured
        state = env.reset()
        assert not np.all(state == 0.0)
        for _ in range(10):
            action = env.action_space.sample()
            next_state, *_ = env.step(action)
            assert not np.all(next_state == 0.0)


if __name__ == "__main__":
    pytest.main()
