import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestDDPG(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        ddpg = A.DDPG(dummy_env)

        assert ddpg.__name__ == 'DDPG'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        ddpg = A.DDPG(dummy_env)

        ddpg.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyContinuous()
        params = A.DDPGParam(batch_size=batch_size)
        ddpg = A.DDPG(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        ddpg.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        ddpg = A.DDPG(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = ddpg.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_target_network_initialization(self):
        dummy_env = E.DummyContinuous()
        ddpg = A.DDPG(dummy_env)

        # Should be initialized to same parameters
        assert self._has_same_parameters(
            ddpg._q.get_parameters(), ddpg._target_q.get_parameters())
        assert self._has_same_parameters(
            ddpg._pi.get_parameters(), ddpg._target_pi.get_parameters())

    def test_update_algorithm_params(self):
        dummy_env = E.DummyContinuous()
        ddpg = A.DDPG(dummy_env)

        tau = 1.0
        gamma = 100.0
        learning_rate = 1e-5
        batch_size = 1000
        start_timesteps = 10
        replay_buffer_size = 100
        param = {'tau': tau,
                 'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size,
                 'start_timesteps': start_timesteps,
                 'replay_buffer_size': replay_buffer_size}

        ddpg.update_algorithm_params(**param)

        assert ddpg._params.tau == tau
        assert ddpg._params.gamma == gamma
        assert ddpg._params.learning_rate == learning_rate
        assert ddpg._params.batch_size == batch_size
        assert ddpg._params.start_timesteps == start_timesteps
        assert ddpg._params.replay_buffer_size == replay_buffer_size

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].data.data, params2[key].data.data):
                return False
        return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
