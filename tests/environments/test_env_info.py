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


if __name__ == "__main__":
    pytest.main()
