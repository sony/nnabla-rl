import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.algorithm import EnvironmentInfo
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestDummy(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscrete()
        dummy = A.Dummy(dummy_env)

        assert dummy.__name__ == 'Dummy'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        dummy.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        experience_num = 100
        fake_states = np.empty(shape=(experience_num, ) +
                               dummy_env.observation_space.shape)
        fake_actions = np.empty(shape=(experience_num, ) +
                                dummy_env.action_space.shape)
        fake_rewards = np.empty(shape=(experience_num, 1))
        fake_non_terminals = np.empty(shape=(experience_num, 1))
        fake_next_states = np.empty(shape=(experience_num, ) +
                                    dummy_env.observation_space.shape)
        fake_next_actions = np.empty(shape=(experience_num, ) +
                                     dummy_env.action_space.shape)

        fake_experiences = zip(fake_states, fake_actions, fake_rewards,
                               fake_non_terminals, fake_next_states, fake_next_actions)
        dummy.train_offline(fake_experiences, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dummy.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape


if __name__ == "__main__":
    pytest.main()
