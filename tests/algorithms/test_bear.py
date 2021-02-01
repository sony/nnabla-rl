import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestBEAR(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        assert bear.__name__ == 'BEAR'

    def test_run_online_training(self):
        """
        Check that error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        with pytest.raises(NotImplementedError):
            bear.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyContinuous()
        params = A.BEARParam(batch_size=batch_size)
        bear = A.BEAR(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        bear.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = bear.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_update_algorithm_params(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        tau = 1.0
        gamma = 0.5
        learning_rate = 1e-5
        batch_size = 1000
        num_q_ensembles = 30
        num_mmd_actions = 5
        num_action_samples = 10
        warmup_iterations = 500
        param = {'tau': tau,
                 'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size,
                 'num_q_ensembles': num_q_ensembles,
                 'num_mmd_actions': num_mmd_actions,
                 'num_action_samples': num_action_samples,
                 'warmup_iterations': warmup_iterations}

        bear.update_algorithm_params(**param)

        assert bear._params.tau == tau
        assert bear._params.gamma == gamma
        assert bear._params.learning_rate == learning_rate
        assert bear._params.batch_size == batch_size
        assert bear._params.num_q_ensembles == num_q_ensembles
        assert bear._params.num_mmd_actions == num_mmd_actions
        assert bear._params.num_action_samples == num_action_samples
        assert bear._params.warmup_iterations == warmup_iterations

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.BEARParam(tau=1.1)
        with pytest.raises(ValueError):
            A.BEARParam(tau=-0.1)
        with pytest.raises(ValueError):
            A.BEARParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.BEARParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.BEARParam(num_q_ensembles=-100)
        with pytest.raises(ValueError):
            A.BEARParam(num_mmd_actions=-100)
        with pytest.raises(ValueError):
            A.BEARParam(num_action_samples=-100)
        with pytest.raises(ValueError):
            A.BEARParam(warmup_iterations=-100)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
