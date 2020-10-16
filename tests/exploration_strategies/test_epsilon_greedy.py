from unittest import mock
import pytest

import numpy as np

from nnabla_rl.exploration_strategies import EpsilonGreedyExplorationStrategy
from nnabla_rl.exploration_strategies.epsilon_greedy import epsilon_greedy_action_selection


class TestEpsilonGreedyExplorationStrategy(object):
    def test_epsilon_greedy_action_selection_always_greedy(self):
        greedy_selector_mock = mock.MagicMock(return_value=1)
        random_selector_mock = mock.MagicMock(return_value=2)

        state = 'test'
        should_be_greedy, is_greedy = epsilon_greedy_action_selection(
            state,
            greedy_selector_mock,
            random_selector_mock,
            epsilon=0.0)
        assert should_be_greedy == 1
        assert is_greedy is True
        greedy_selector_mock.assert_called_once()

    def test_epsilon_greedy_action_selection_always_random(self):
        greedy_selector_mock = mock.MagicMock(return_value=1)
        random_selector_mock = mock.MagicMock(return_value=2)

        state = 'test'
        should_be_random, is_greedy = epsilon_greedy_action_selection(
            state,
            greedy_selector_mock,
            random_selector_mock,
            epsilon=1.0)
        assert should_be_random == 2
        assert is_greedy is False
        random_selector_mock.assert_called_once()

    def test_epsilon_greedy_action_selection(self):
        greedy_selector_mock = mock.MagicMock(return_value=1)
        random_selector_mock = mock.MagicMock(return_value=2)

        state = 'test'
        action, is_greedy = epsilon_greedy_action_selection(
            state,
            greedy_selector_mock,
            random_selector_mock,
            epsilon=0.5)
        if is_greedy:
            assert action == 1
            greedy_selector_mock.assert_called_once()
        else:
            assert action == 2
            random_selector_mock.assert_called_once()

    def test_compute_epsilon(self):
        initial_epsilon = 1.0
        final_epsilon = 0.1
        max_explore_steps = 100
        greedy_selector_mock = mock.MagicMock(return_value=1)
        random_selector_mock = mock.MagicMock(return_value=2)
        explorer = EpsilonGreedyExplorationStrategy(initial_epsilon,
                                                    final_epsilon,
                                                    max_explore_steps,
                                                    greedy_selector_mock,
                                                    random_selector_mock)

        def expected_epsilon(step):
            epsilon = initial_epsilon - \
                (initial_epsilon - final_epsilon) / max_explore_steps * step
            return max(epsilon, final_epsilon)

        assert np.isclose(explorer._compute_epsilon(1), expected_epsilon(1))
        assert np.isclose(explorer._compute_epsilon(50), expected_epsilon(50))
        assert np.isclose(explorer._compute_epsilon(99), expected_epsilon(99))
        assert np.isclose(explorer._compute_epsilon(100),
                          expected_epsilon(100))
        assert explorer._compute_epsilon(200) == final_epsilon


if __name__ == '__main__':
    pytest.main()
