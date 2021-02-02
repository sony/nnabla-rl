import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestMunchausenDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.MunchausenDQN(dummy_env)

        assert dqn.__name__ == 'MunchausenDQN'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        params = A.MunchausenDQNParam()
        params.start_timesteps = 5
        params.batch_size = 5
        params.learner_update_frequency = 1
        params.target_update_frequency = 1
        dqn = A.MunchausenDQN(dummy_env, params=params)

        dqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        params = A.MunchausenDQNParam()
        params.batch_size = batch_size
        params.learner_update_frequency = 1
        params.target_update_frequency = 1

        dqn = A.MunchausenDQN(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        dqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.MunchausenDQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(batch_size=-1)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(learner_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(target_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(start_timesteps=-1000)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(replay_buffer_size=-1000)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(initial_epsilon=1.5)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(final_epsilon=1.1)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(test_epsilon=-1000)
        with pytest.raises(ValueError):
            A.MunchausenDQNParam(max_explore_steps=-100)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
