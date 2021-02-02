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
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyContinuous()
        batch_size = 5
        params = A.DDPGParam(batch_size=batch_size, start_timesteps=5)
        ddpg = A.DDPG(dummy_env, params=params)

        ddpg.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
