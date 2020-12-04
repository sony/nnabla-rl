import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestQRDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        qrdqn = A.QRDQN(dummy_env)

        assert qrdqn.__name__ == 'QRDQN'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyDiscreteImg()
        params = A.QRDQNParam()
        params.start_timesteps = 5
        params.batch_size = 5
        params.learner_update_frequency = 1
        params.target_update_frequency = 1
        qrdqn = A.QRDQN(dummy_env, params=params)

        qrdqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        params = A.QRDQNParam()
        params.batch_size = batch_size
        params.learner_update_frequency = 1
        params.target_update_frequency = 1

        qrdqn = A.QRDQN(dummy_env, params=params)

        buffer = ReplayBuffer()
        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        qrdqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        qrdqn = A.QRDQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = qrdqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_update_algorithm_params(self):
        dummy_env = E.DummyDiscreteImg()
        qrdqn = A.QRDQN(dummy_env)

        gamma = 0.5
        learning_rate = 1e-5
        batch_size = 1000
        param = {'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size}

        qrdqn.update_algorithm_params(**param)

        assert qrdqn._params.gamma == gamma
        assert qrdqn._params.learning_rate == learning_rate
        assert qrdqn._params.batch_size == batch_size

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.QRDQNParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.QRDQNParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.QRDQNParam(batch_size=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(learner_update_frequency=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(max_explore_steps=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(learning_rate=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(initial_epsilon=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(final_epsilon=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(test_epsilon=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(num_quantiles=-1)
        with pytest.raises(ValueError):
            A.QRDQNParam(kappa=-1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
