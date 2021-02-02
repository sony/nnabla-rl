import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestTD3(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        td3 = A.TD3(dummy_env)

        assert td3.__name__ == 'TD3'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyContinuous()
        batch_size = 5
        params = A.TD3Param(batch_size=batch_size, start_timesteps=5)
        td3 = A.TD3(dummy_env, params=params)

        td3.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyContinuous()
        batch_size = 5
        params = A.TD3Param(batch_size=batch_size)
        td3 = A.TD3(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        td3.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        td3 = A.TD3(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = td3.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_params_lie_in_range(self):
        with pytest.raises(ValueError):
            A.TD3Param(d=0)
        with pytest.raises(ValueError):
            A.TD3Param(d=-1)
        with pytest.raises(ValueError):
            A.TD3Param(tau=-0.5)
        with pytest.raises(ValueError):
            A.TD3Param(tau=100.0)
        with pytest.raises(ValueError):
            A.TD3Param(gamma=-100.0)
        with pytest.raises(ValueError):
            A.TD3Param(gamma=10.0)
        with pytest.raises(ValueError):
            A.TD3Param(exploration_noise_sigma=-1.0)
        with pytest.raises(ValueError):
            A.TD3Param(train_action_noise_sigma=-1.0)
        with pytest.raises(ValueError):
            A.TD3Param(train_action_noise_abs=-1.0)
        with pytest.raises(ValueError):
            A.TD3Param(batch_size=-1)
        with pytest.raises(ValueError):
            A.TD3Param(start_timesteps=-1)
        with pytest.raises(ValueError):
            A.TD3Param(replay_buffer_size=-1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
