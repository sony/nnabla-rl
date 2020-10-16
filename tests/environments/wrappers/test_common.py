import pytest

import numpy as np

from nnabla_rl.environments.wrappers.common import NumpyFloat32Env
import nnabla_rl.environments as E


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
