import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestCategoricalDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        categorical_dqn = A.CategoricalDQN(dummy_env)

        assert categorical_dqn.__name__ == 'CategoricalDQN'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyDiscreteImg()
        params = A.CategoricalDQNParam()
        params.start_timesteps = 5
        params.batch_size = 5
        params.learner_update_frequency = 1
        params.target_update_frequency = 1
        categorical_dqn = A.CategoricalDQN(dummy_env, params=params)

        categorical_dqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyDiscreteImg()
        params = A.CategoricalDQNParam(batch_size=batch_size)
        categorical_dqn = A.CategoricalDQN(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        categorical_dqn.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        categorical_dqn = A.CategoricalDQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = categorical_dqn.compute_eval_action(state)

        assert action.shape == (1, )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
