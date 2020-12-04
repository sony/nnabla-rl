import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestBCQ(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        bcq = A.BCQ(dummy_env)

        assert bcq.__name__ == 'BCQ'

    def test_run_online_training(self):
        """
        Check that error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        params = A.BCQParam()
        bcq = A.BCQ(dummy_env, params=params)

        with pytest.raises(NotImplementedError):
            bcq.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyContinuous()
        params = A.BCQParam(batch_size=batch_size)
        bcq = A.BCQ(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        bcq.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        bcq = A.BCQ(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = bcq.compute_eval_action(state)
        assert action.shape == dummy_env.action_space.shape

    def test_update_algorithm_params(self):
        dummy_env = E.DummyContinuous()
        bcq = A.BCQ(dummy_env)

        tau = 1.0
        gamma = 0.5
        learning_rate = 1e-5
        lmb = 0.1
        phi = 0.2
        batch_size = 1000
        num_q_ensembles = 30
        num_action_samples = 10
        param = {'tau': tau,
                 'gamma': gamma,
                 'learning_rate': learning_rate,
                 'lmb': lmb,
                 'phi': phi,
                 'batch_size': batch_size,
                 'num_q_ensembles': num_q_ensembles,
                 'num_action_samples': num_action_samples}

        bcq.update_algorithm_params(**param)

        assert bcq._params.tau == tau
        assert bcq._params.gamma == gamma
        assert bcq._params.learning_rate == learning_rate
        assert bcq._params.lmb == lmb
        assert bcq._params.phi == phi
        assert bcq._params.batch_size == batch_size
        assert bcq._params.num_q_ensembles == num_q_ensembles
        assert bcq._params.num_action_samples == num_action_samples

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.BCQParam(tau=1.1)
        with pytest.raises(ValueError):
            A.BCQParam(tau=-0.1)
        with pytest.raises(ValueError):
            A.BCQParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.BCQParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.BCQParam(lmb=-0.1)
        with pytest.raises(ValueError):
            A.BCQParam(phi=-0.1)
        with pytest.raises(ValueError):
            A.BCQParam(num_q_ensembles=-100)
        with pytest.raises(ValueError):
            A.BCQParam(num_action_samples=-100)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
