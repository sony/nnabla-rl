import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.algorithm import EnvironmentInfo
from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        assert dqn.__name__ == 'DQN'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """
        dummy_env = E.DummyDiscreteImg()
        params = A.DQNParam()
        params.start_timesteps = 5
        params.batch_size = 5
        params.learner_update_frequency = 1
        params.target_update_frequency = 1
        dqn = A.DQN(dummy_env, params=params)

        dqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        params = A.DQNParam()
        params.batch_size = batch_size
        params.learner_update_frequency = 1
        params.target_update_frequency = 1

        dqn = A.DQN(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        dqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_update_algorithm_params(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        gamma = 0.1
        batch_size = 10
        learning_rate = 0.1
        decay = 1.5
        momentum = 0.5
        min_squared_gradient = 0.01
        learner_update_frequency = 100
        target_update_frequency = 5000
        start_timesteps = 50
        replay_buffer_size = 1000
        initial_epsilon = 0.5
        final_epsilon = 0.1
        test_epsilon = 0.1
        max_explore_steps = 10

        param = {'gamma': gamma,
                 'batch_size': batch_size,
                 'learning_rate': learning_rate,
                 'decay': decay,
                 'momentum': momentum,
                 'min_squared_gradient': min_squared_gradient,
                 'learner_update_frequency': learner_update_frequency,
                 'target_update_frequency': target_update_frequency,
                 'start_timesteps': start_timesteps,
                 'replay_buffer_size': replay_buffer_size,
                 'initial_epsilon': initial_epsilon,
                 'final_epsilon': final_epsilon,
                 'test_epsilon': test_epsilon,
                 'max_explore_steps': max_explore_steps
                 }

        dqn.update_algorithm_params(**param)

        assert dqn._params.gamma == gamma
        assert dqn._params.batch_size == batch_size
        assert dqn._params.learning_rate == learning_rate
        assert dqn._params.min_squared_gradient == min_squared_gradient
        assert dqn._params.learner_update_frequency == learner_update_frequency
        assert dqn._params.target_update_frequency == target_update_frequency
        assert dqn._params.start_timesteps == start_timesteps
        assert dqn._params.replay_buffer_size == replay_buffer_size
        assert dqn._params.initial_epsilon == initial_epsilon
        assert dqn._params.final_epsilon == final_epsilon
        assert dqn._params.test_epsilon == test_epsilon
        assert dqn._params.max_explore_steps == max_explore_steps

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.DQNParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.DQNParam(batch_size=-1)
        with pytest.raises(ValueError):
            A.DQNParam(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.DQNParam(min_squared_gradient=-0.1)
        with pytest.raises(ValueError):
            A.DQNParam(learner_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.DQNParam(target_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.DQNParam(start_timesteps=-1000)
        with pytest.raises(ValueError):
            A.DQNParam(replay_buffer_size=-1000)
        with pytest.raises(ValueError):
            A.DQNParam(initial_epsilon=1.5)
        with pytest.raises(ValueError):
            A.DQNParam(final_epsilon=1.1)
        with pytest.raises(ValueError):
            A.DQNParam(test_epsilon=-1000)
        with pytest.raises(ValueError):
            A.DQNParam(max_explore_steps=-100)

    def test_solver_has_correct_parameters(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        q_solver_parms = dqn._solvers()["q_solver"].get_parameters()
        q_network_params = dqn._q.get_parameters()
        target_q_network_params = dqn._target_q.get_parameters()

        assert is_same_parameter_id_and_key(q_network_params, q_solver_parms)
        assert not is_same_parameter_id_and_key(
            target_q_network_params, q_solver_parms)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences, is_same_parameter_id_and_key
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences, is_same_parameter_id_and_key
