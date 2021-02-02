import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestMunchausenIQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        m_iqn = A.MunchausenIQN(dummy_env)

        assert m_iqn.__name__ == 'MunchausenIQN'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        params = A.MunchausenIQNParam()
        params.start_timesteps = 5
        params.batch_size = 5
        params.learner_update_frequency = 1
        params.target_update_frequency = 1
        m_iqn = A.MunchausenIQN(dummy_env, params=params)

        m_iqn.train_online(dummy_env, total_iterations=5)

    def test_run_offline_training(self):
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        params = A.MunchausenIQNParam()
        params.batch_size = batch_size
        params.learner_update_frequency = 1
        params.target_update_frequency = 1

        m_iqn = A.MunchausenIQN(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        m_iqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        m_iqn = A.MunchausenIQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = m_iqn.compute_eval_action(state)

        assert action.shape == (1, )

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(batch_size=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(learner_update_frequency=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(max_explore_steps=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(learning_rate=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(initial_epsilon=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(final_epsilon=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(test_epsilon=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(K=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(N=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(N_prime=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(kappa=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(embedding_dim=-1)
        with pytest.raises(ValueError):
            A.MunchausenIQNParam(clipping_value=100)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
