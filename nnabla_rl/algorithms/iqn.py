import nnabla as nn

import nnabla.functions as F
import nnabla.solvers as S

import numpy as np

from collections import namedtuple
from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.exploration_strategies.epsilon_greedy import epsilon_greedy_action_selection
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.data import marshall_experiences
import nnabla_rl.exploration_strategies as ES
import nnabla_rl.models as M
import nnabla_rl.functions as RF


def default_quantile_function_builder(scope_name, state_dim, action_dim, embedding_dim):
    return M.IQNQuantileFunction(scope_name, state_dim, action_dim, embedding_dim)


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


def risk_neutral_measure(tau):
    return tau


@dataclass
class IQNParam(AlgorithmParam):
    batch_size: int = 32
    gamma: float = 0.99
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    max_explore_steps: int = 1000000
    learning_rate: float = 0.00005
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    N: int = 64
    N_prime: int = 64
    K: int = 32
    kappa: float = 1.0
    embedding_dim: int = 64

    def __post_init__(self):
        '''__post_init__

        Check that set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.learner_update_frequency,
                              'learner_update_frequency')
        self._assert_positive(self.target_update_frequency,
                              'target_update_frequency')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.initial_epsilon, 'initial_epsilon')
        self._assert_positive(self.final_epsilon, 'final_epsilon')
        self._assert_positive(self.test_epsilon, 'test_epsilon')
        self._assert_positive(self.N, 'N')
        self._assert_positive(self.N_prime, 'N_prime')
        self._assert_positive(self.K, 'K')
        self._assert_positive(self.kappa, 'kappa')
        self._assert_positive(self.embedding_dim, 'embedding_dim')


class IQN(Algorithm):
    '''Implicit Quantile Network algorithm implementation.

    This class implements the Implicit Quantile Network (IQN) algorithm
    proposed by W. Dabney, et al. in the paper: "Implicit Quantile Networks for Distributional Reinforcement Learning"
    For detail see: https://arxiv.org/pdf/1806.06923.pdf
    '''

    def __init__(self, env_info,
                 quantile_function_builder=default_quantile_function_builder,
                 risk_measure_function=risk_neutral_measure,
                 params=IQNParam(),
                 replay_buffer_builder=default_replay_buffer_builder):
        super(IQN, self).__init__(env_info, params=params)

        if not self._env_info.is_discrete_action_env():
            raise ValueError(
                '{} only supports discrete action environment'.format(self.__name__))
        state_shape = self._env_info.observation_space.shape
        self._n_action = self._env_info.action_space.n

        self._quantile_function = quantile_function_builder(
            'quantile_function', state_shape, self._n_action, self._params.embedding_dim)
        assert isinstance(self._quantile_function,
                          M.StateActionQuantileFunction)

        self._target_quantile_function = quantile_function_builder(
            'target_quantile_function', state_shape, self._n_action, self._params.embedding_dim)
        assert isinstance(self._target_quantile_function,
                          M.StateActionQuantileFunction)

        self._risk_measure_function = risk_measure_function

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
            nn.Variable((params.batch_size, *state_shape))
        a_current_var = \
            nn.Variable((params.batch_size, 1))
        reward_var = nn.Variable((params.batch_size, 1))
        non_terminal_var = nn.Variable((params.batch_size, 1))
        s_next_var = nn.Variable((params.batch_size, *state_shape))

        TrainingVariables = namedtuple(
            'TrainingVariables', ['s_current', 'a_current', 'reward', 'non_terminal', 's_next'])
        self._training_variables = \
            TrainingVariables(s_current_var, a_current_var, reward_var,
                              non_terminal_var, s_next_var)

        # Training loss/output
        self._quantile_huber_loss = None

        # Evaluation input variables
        s_eval_var = nn.Variable((1, *state_shape))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['s_eval'])
        self._evaluation_variables = EvaluationVariables(s_eval_var)

        # Evaluation output
        self._a_greedy = None

    def _post_init(self):
        super(IQN, self)._post_init()
        copy_network_parameters(
            self._quantile_function.get_parameters(),
            self._target_quantile_function.get_parameters(),
            tau=1.0)

    def compute_eval_action(self, state):
        action, _ = epsilon_greedy_action_selection(state,
                                                    self._greedy_action_selector,
                                                    self._random_action_selector,
                                                    epsilon=self._params.test_epsilon)
        return action

    def _build_training_graph(self):
        tau_k = self._risk_measure_function(
            self._sample_tau(shape=(self._params.batch_size, self._params.K)))
        policy_quantiles = self._target_quantile_function.quantiles(
            self._training_variables.s_next, tau_k)
        a_star = self._compute_argmax_q(policy_quantiles)

        tau_j = self._sample_tau(
            shape=(self._params.batch_size, self._params.N_prime))
        target_quantiles = self._target_quantile_function.quantiles(
            self._training_variables.s_next, tau_j)
        Z_tau_j = self._quantiles_of(target_quantiles, a_star)
        assert Z_tau_j.shape == (self._params.batch_size,
                                 self._params.N_prime)
        target = self._training_variables.reward + \
            self._training_variables.non_terminal * self._params.gamma * Z_tau_j
        target = RF.expand_dims(target, axis=1)
        target.need_grad = False
        assert target.shape == (self._params.batch_size,
                                1,
                                self._params.N_prime)

        tau_i = self._sample_tau(
            shape=(self._params.batch_size, self._params.N))
        quantiles = self._quantile_function.quantiles(
            self._training_variables.s_current, tau_i)
        Z_tau_i = self._quantiles_of(
            quantiles, self._training_variables.a_current)
        Z_tau_i = RF.expand_dims(Z_tau_i, axis=2)
        tau_i = RF.expand_dims(tau_i, axis=2)
        assert Z_tau_i.shape == (self._params.batch_size,
                                 self._params.N,
                                 1)
        assert tau_i.shape == Z_tau_i.shape

        quantile_huber_loss = RF.quantile_huber_loss(
            target, Z_tau_i, self._params.kappa, tau_i)
        assert quantile_huber_loss.shape == (self._params.batch_size,
                                             self._params.N,
                                             self._params.N_prime)
        quantile_huber_loss = F.mean(quantile_huber_loss, axis=2)
        quantile_huber_loss = F.sum(quantile_huber_loss, axis=1)
        self._quantile_huber_loss = F.mean(quantile_huber_loss)

    def _build_evaluation_graph(self):
        tau = self._risk_measure_function(
            self._sample_tau(shape=(1, self._params.K)))
        quantiles = self._quantile_function.quantiles(
            self._evaluation_variables.s_eval, tau)
        self._a_greedy = self._compute_argmax_q(quantiles)

    def _setup_solver(self):
        self._quantile_function_solver = S.Adam(
            alpha=self._params.learning_rate, eps=1e-2 / self._params.batch_size)
        self._quantile_function_solver.set_parameters(
            self._quantile_function.get_parameters())

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
            self._iqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._iqn_training(buffer)

    def _iqn_training(self, replay_buffer):
        if self.iteration_num % self._params.learner_update_frequency != 0:
            return

        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)

        self._training_variables.s_current.d = s
        self._training_variables.a_current.d = a
        self._training_variables.reward.d = r
        self._training_variables.non_terminal.d = non_terminal
        self._training_variables.s_next.d = s_next

        self._quantile_function_solver.zero_grad()
        self._quantile_huber_loss.forward()
        self._quantile_huber_loss.backward()
        self._quantile_function_solver.update()

        # Update target net
        if self.iteration_num % self._params.target_update_frequency == 0:
            copy_network_parameters(
                self._quantile_function.get_parameters(),
                self._target_quantile_function.get_parameters(),
                tau=1.0)

    def _greedy_action_selector(self, s):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._a_greedy.forward()
        return self._a_greedy.d

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, ))

    def _compute_argmax_q(self, quantiles):
        q_values = self._compute_q_values(quantiles)
        return RF.argmax(q_values, axis=1)

    def _compute_q_values(self, quantiles):
        batch_size = quantiles.shape[0]
        assert len(quantiles.shape) == 3
        assert quantiles.shape[2] == self._n_action
        quantiles = F.transpose(quantiles, axes=(0, 2, 1))
        q_values = F.mean(quantiles, axis=2)
        assert q_values.shape == (batch_size, self._n_action)
        return q_values

    def _quantiles_of(self, quantiles, a):
        one_hot = self._to_one_hot(a, shape=quantiles.shape)
        quantiles = quantiles * one_hot
        quantiles = F.sum(quantiles, axis=2)
        assert len(quantiles.shape) == 2

        return quantiles

    def _sample_tau(self, shape):
        return F.rand(low=0.0, high=1.0, shape=shape)

    def _to_one_hot(self, a, shape):
        batch_size = a.shape[0]
        a = F.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = F.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = F.broadcast(one_hot, shape=shape)
        return one_hot

    def _models(self):
        models = {}
        models[self._quantile_function.scope_name] = self._quantile_function
        models[self._target_quantile_function.scope_name] = self._target_quantile_function
        return models

    def _solvers(self):
        solvers = {}
        solvers['quantile_function_solver'] = self._quantile_function_solver
        return solvers
