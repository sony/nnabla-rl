import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.models.atari.distributional_functions import C51ValueDistributionFunction
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

        assert action.shape == (1,)

    def test_probabilities_of(self):
        dummy_env = E.DummyDiscreteImg()
        n_atoms = 10
        n_action = dummy_env.action_space.n
        params = A.CategoricalDQNParam(num_atoms=n_atoms)
        categorical_dqn = A.CategoricalDQN(dummy_env, params=params)

        state_shape = (4, 84, 84)
        scope_name = "test"
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             state_shape=state_shape,
                                             num_actions=n_action,
                                             num_atoms=n_atoms)

        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        probabilities = model.probabilities(input_state)

        actions = nn.Variable.from_numpy_array(
            np.asarray([[3]]))
        action_probabilities = categorical_dqn._probabilities_of(
            probabilities, actions)
        action_probabilities.forward()

        assert action_probabilities.shape == (1, n_atoms)
        assert np.allclose(probabilities.d[0][3], action_probabilities.d[0])

    def test_to_one_hot(self):
        dummy_env = E.DummyDiscreteImg()
        n_action = dummy_env.action_space.n
        n_atoms = 10
        params = A.CategoricalDQNParam(num_atoms=n_atoms)
        categorical_dqn = A.CategoricalDQN(dummy_env, params=params)

        actions = nn.Variable.from_numpy_array(
            np.asarray([[0], [1], [2], [3]]))

        assert actions.shape == (4, 1)

        val = categorical_dqn._to_one_hot(actions)
        val.forward()

        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected = np.expand_dims(expected, axis=1)
        expected = np.broadcast_to(expected, shape=(4, n_atoms, n_action))

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
