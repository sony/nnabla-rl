# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import numpy as np
import pytest

from nnabla_rl.environment_explorers.epsilon_greedy_explorer import (LinearDecayEpsilonGreedyExplorer,
                                                                     LinearDecayEpsilonGreedyExplorerConfig,
                                                                     epsilon_greedy_action_selection)


class TestEpsilonGreedyActionStrategy(object):
    def test_epsilon_greedy_action_selection_always_greedy(self):
        greedy_selector_mock = mock.MagicMock(return_value=(1, {}))
        random_selector_mock = mock.MagicMock(return_value=(2, {}))

        state = 'test'
        (should_be_greedy, _), is_greedy = epsilon_greedy_action_selection(
            state,
            greedy_selector_mock,
            random_selector_mock,
            epsilon=0.0)
        assert should_be_greedy == 1
        assert is_greedy is True
        greedy_selector_mock.assert_called_once()

    def test_epsilon_greedy_action_selection_always_random(self):
        greedy_selector_mock = mock.MagicMock(return_value=(1, {}))
        random_selector_mock = mock.MagicMock(return_value=(2, {}))

        state = 'test'
        (should_be_random, _), is_greedy = epsilon_greedy_action_selection(
            state,
            greedy_selector_mock,
            random_selector_mock,
            epsilon=1.0)
        assert should_be_random == 2
        assert is_greedy is False
        random_selector_mock.assert_called_once()

    def test_epsilon_greedy_action_selection(self):
        greedy_selector_mock = mock.MagicMock(return_value=(1, {}))
        random_selector_mock = mock.MagicMock(return_value=(2, {}))

        state = 'test'
        (action, _), is_greedy = epsilon_greedy_action_selection(
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
        greedy_selector_mock = mock.MagicMock(return_value=(1, {}))
        random_selector_mock = mock.MagicMock(return_value=(2, {}))
        config = LinearDecayEpsilonGreedyExplorerConfig(initial_epsilon=initial_epsilon,
                                                        final_epsilon=final_epsilon,
                                                        max_explore_steps=max_explore_steps)
        explorer = LinearDecayEpsilonGreedyExplorer(greedy_selector_mock,
                                                    random_selector_mock,
                                                    env_info=None,
                                                    config=config)

        def expected_epsilon(step):
            epsilon = initial_epsilon - \
                (initial_epsilon - final_epsilon) / max_explore_steps * step
            return max(epsilon, final_epsilon)

        assert np.isclose(explorer._compute_epsilon(1), expected_epsilon(1))
        assert np.isclose(explorer._compute_epsilon(50), expected_epsilon(50))
        assert np.isclose(explorer._compute_epsilon(99), expected_epsilon(99))
        assert np.isclose(explorer._compute_epsilon(100), expected_epsilon(100))
        assert explorer._compute_epsilon(200) == final_epsilon


if __name__ == '__main__':
    pytest.main()
