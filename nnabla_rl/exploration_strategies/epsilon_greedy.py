import numpy as np

from nnabla_rl.exploration_strategy import ExplorationStrategy


def epsilon_greedy_action_selection(state, greedy_action_selector, random_action_selector, epsilon):
    if np.random.rand() > epsilon:
        # optimal action
        return greedy_action_selector(state), True
    else:
        # random action
        return random_action_selector(state), False


class EpsilonGreedyExplorationStrategy(ExplorationStrategy):
    def __init__(self,
                 initial_epsilon,
                 final_epsilon,
                 max_explore_steps,
                 greedy_action_selector,
                 random_action_selector):
        super(EpsilonGreedyExplorationStrategy, self).__init__()
        self._initial_epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._max_explore_steps = max_explore_steps
        self._greedy_action_selector = greedy_action_selector
        self._random_action_selector = random_action_selector

    def select_action(self, step, state):
        epsilon = self._compute_epsilon(step)
        action, _ = epsilon_greedy_action_selection(state,
                                                    self._greedy_action_selector,
                                                    self._random_action_selector,
                                                    epsilon)
        return action

    def _compute_epsilon(self, step):
        assert 0 <= step
        delta_epsilon = step / self._max_explore_steps \
            * (self._initial_epsilon - self._final_epsilon)
        epsilon = self._initial_epsilon - delta_epsilon
        return max(epsilon, self._final_epsilon)
