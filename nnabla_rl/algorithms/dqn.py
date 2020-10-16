import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S

import warnings
import random
from collections import namedtuple
from dataclasses import dataclass

import gym
import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.debugging import print_network, view_graph
from nnabla_rl.utils.copy import copy_network_parameters
import nnabla_rl.exploration_strategies as ES
from nnabla_rl.exploration_strategies.epsilon_greedy import epsilon_greedy_action_selection
import nnabla_rl.functions as RF
import nnabla_rl.models as M


def default_q_func_builder(scope_name, state_shape, n_action):
    return M.DQNQFunction(scope_name, state_shape, n_action)


def default_q_solver_builder(q_func, params):
    try:
        solver = S.RMSpropgraves(
            lr=params.learning_rate, decay=params.decay,
            momentum=params.momentum, eps=params.min_squared_gradient)
    except:
        warnings.warn("Instead of RMSpropgraves, use Adam as a Solver, \
            Please check learning rate. It might be needed to tune it")
        solver = S.Adam(params.learning_rate)
    solver.set_parameters(q_func.get_parameters())
    return solver


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


@dataclass
class DQNParam(AlgorithmParam):
    gamma: float = 0.99
    batch_size: int = 32
    # optimizer
    learning_rate: float = 2.5e-4
    # this decay is equivalent to 'gradient momentum' and 'squared gradient momentum' of the nature paper
    decay: float = 0.95
    momentum: float = 0.0
    min_squared_gradient: float = 0.01
    # network update
    learner_update_frequency: float = 4
    target_update_frequency: float = 10000
    # buffers
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    # explore
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.1
    test_epsilon: float = 0.05
    max_explore_steps: int = 1000000

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 <= self.batch_size):
            raise ValueError('batch size must not be negative')
        if not (0 <= self.learning_rate):
            raise ValueError('learning rate must not be negative')
        if not (0 <= self.decay):
            raise ValueError('decay must not be negative')
        if not (0 <= self.min_squared_gradient):
            raise ValueError('min_squared_gradient must not be negative')
        if not (0 <= self.learner_update_frequency):
            raise ValueError('learner update frequency must not be negative')
        if not (0 <= self.target_update_frequency):
            raise ValueError('target update frequency must not be negative')
        if self.start_timesteps is not None:
            if not (0 <= self.start_timesteps):
                raise ValueError('start timesteps must not be negative')
        if (self.start_timesteps > self.replay_buffer_size):
            raise ValueError('start timesteps should be smaller than \
                replay buffer size')
        if not (0 <= self.replay_buffer_size):
            raise ValueError('replay buffer size must not be negative')
        if not ((0.0 <= self.initial_epsilon) & (self.initial_epsilon <= 1.0)):
            raise ValueError('initial epsilon must lie between [0.0, 1.0]')
        if not ((0.0 <= self.final_epsilon) & (self.final_epsilon <= 1.0)):
            raise ValueError('final epsilon must lie between [0.0, 1.0]')
        if not ((0.0 <= self.test_epsilon) & (self.test_epsilon <= 1.0)):
            raise ValueError('test epsilon must lie between [0.0, 1.0]')
        if not (0 <= self.max_explore_steps):
            raise ValueError('max explore step must not be negative')


class DQN(Algorithm):
    def __init__(self, env_info,
                 q_func_builder=default_q_func_builder,
                 params=DQNParam(),
                 replay_buffer_builder=default_replay_buffer_builder):
        super(DQN, self).__init__(env_info, params=params)

        if not isinstance(env_info.action_space, gym.spaces.Discrete):
            raise ValueError('Invalid env_info Action space of DQN must be {}'
                             .format(gym.spaces.Discrete))

        _state_shape = env_info.observation_space.shape
        _n_action = env_info.action_space.n  # discrete

        self._q = q_func_builder(
            scope_name='q',
            state_shape=_state_shape,
            n_action=_n_action)
        assert isinstance(self._q, M.QFunction)

        self._target_q = q_func_builder(
            scope_name='target_q',
            state_shape=_state_shape,
            n_action=_n_action)
        assert isinstance(self._target_q, M.QFunction)

        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = replay_buffer_builder(
            capacity=params.replay_buffer_size)

        self._exploration_strategy = ES.EpsilonGreedyExplorationStrategy(self._params.initial_epsilon,
                                                                         self._params.final_epsilon,
                                                                         self._params.max_explore_steps,
                                                                         self._greedy_action_selector,
                                                                         self._random_action_selector)

        # Training input variables
        s_current_var = \
            nn.Variable((params.batch_size, *_state_shape))
        a_current_var = \
            nn.Variable((params.batch_size, 1)
                        )  # discrete (having 1 action dim)
        s_next_var = nn.Variable((params.batch_size, *_state_shape))
        reward_var = nn.Variable((params.batch_size, 1))
        non_terminal_var = nn.Variable((params.batch_size, 1))
        weight_var = nn.Variable((params.batch_size, 1))
        TrainingVariables = namedtuple(
            'TrainingVariables',
            ['s_current', 'a_current', 'reward', 's_next', 'non_terminal', "weight"])
        self._training_variables = \
            TrainingVariables(s_current_var, a_current_var, reward_var,
                              s_next_var, non_terminal_var, weight_var)

        # Training loss/output
        self._q_var = None
        self._huber_loss_var = None
        self._td_error_var = None

        # Evaluation input variables
        s_eval_var = nn.Variable((1, *_state_shape))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['s_eval'])
        self._evaluation_variables = EvaluationVariables(s_eval_var)

        # Evaluation output
        self._a_greedy = None

    def _post_init(self):
        super(DQN, self)._post_init()
        copy_network_parameters(
            self._q.get_parameters(), self._target_q.get_parameters(), tau=1.0)

    def compute_eval_action(self, s):
        action, _ = epsilon_greedy_action_selection(s,
                                                    self._greedy_action_selector,
                                                    self._random_action_selector,
                                                    epsilon=self._params.test_epsilon)
        return action

    def _build_training_graph(self):
        # target q
        max_q_var = self._target_q.maximum(self._training_variables.s_next)
        target_q_var = \
            self._training_variables.reward \
            + self._params.gamma \
            * self._training_variables.non_terminal \
            * max_q_var

        target_q_var.need_grad = False

        # predict q
        self._q_var = self._q.q(self._training_variables.s_current,
                                self._training_variables.a_current)

        # take loss
        self._huber_loss_var = F.sum(0.5
                                     * F.huber_loss(self._q_var, target_q_var))

        # compute td_error for prioritized replay buffer
        self._td_error_var = F.absolute_error(self._q_var, target_q_var)
        self._td_error_var.need_grad = False

    def _build_evaluation_graph(self):
        self._a_greedy = self._q.argmax(self._evaluation_variables.s_eval)

    def _setup_solver(self):
        self._q_solver = default_q_solver_builder(self._q, self._params)

    def _run_online_training_iteration(self, env):
        if self._state is None:
            self._state = env.reset()

        if self.iteration_num < self._params.start_timesteps:
            self._action = self._random_action_selector(self._state)
        else:
            self._action = self._exploration_strategy.select_action(
                self.iteration_num, self._state)

        self._next_state, r, done, _ = env.step(self._action)
        non_terminal = np.float32(0.0 if done else 1.0)
        experience = \
            (self._state, self._action, [r], [non_terminal], self._next_state)
        self._replay_buffer.append(experience)

        if done:
            self._state = env.reset()
        else:
            self._state = self._next_state

        if self._params.start_timesteps < self.iteration_num:
            self._dqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._dqn_training(buffer)

    def _greedy_action_selector(self, s):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._a_greedy.forward()
        return self._a_greedy.d

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, ))

    def _dqn_training(self, replay_buffer):
        if self.iteration_num % self._params.learner_update_frequency != 0:
            return

        experiences, info = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)

        self._training_variables.s_current.d = s
        self._training_variables.a_current.d = a
        self._training_variables.s_next.d = s_next
        self._training_variables.reward.d = r
        self._training_variables.non_terminal.d = non_terminal
        self._training_variables.weight.d = info["weights"]

        # update priority
        self._td_error_var.forward(clear_no_need_grad=True)
        errors = self._td_error_var.d.copy()
        replay_buffer.update_priorities(errors)

        # update model
        self._huber_loss_var.forward(clear_no_need_grad=True)
        self._q_solver.zero_grad()
        self._huber_loss_var.backward(clear_buffer=True)
        self._q_solver.update()

        # Update target net
        if self.iteration_num % self._params.target_update_frequency == 0:
            copy_network_parameters(
                self._q.get_parameters(),
                self._target_q.get_parameters(), tau=1.0)

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        models[self._target_q.scope_name] = self._target_q
        return models

    def _solvers(self):
        solvers = {}
        solvers['q_solver'] = self._q_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}

        latest_iteration_state['scalar']['predicted_q_val'] = self._q_var.d.flatten(
        )
        latest_iteration_state['scalar']['td_loss'] = self._huber_loss_var.d.flatten(
        )
        return latest_iteration_state
