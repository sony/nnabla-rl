import nnabla as nn

import nnabla.functions as NF
import nnabla.solvers as NS

import numpy as np

from collections import namedtuple
from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.exploration_strategies.epsilon_greedy import epsilon_greedy_action_selection
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.logger import logger
import nnabla_rl.exploration_strategies as ES
import nnabla_rl.models as M
import nnabla_rl.functions as RF


def default_quantile_dist_function_builder(scope_name, state_dim, action_dim, num_quantiles):
    return M.QRDQNQuantileDistributionFunction(scope_name, state_dim, action_dim, num_quantiles)


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


@dataclass
class QRDQNParam(AlgorithmParam):
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
    num_quantiles: int = 200
    kappa: float = 1.0

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
        self._assert_positive(self.num_quantiles, 'num_quantiles')
        self._assert_positive(self.kappa, 'kappa')


class QRDQN(Algorithm):
    '''Quantile Regression DQN algorithm implementation.

    This class implements the Quantile Regression DQN algorithm
    proposed by W. Dabney, et al. in the paper: "Distributional Reinforcement Learning with Quantile Regression"
    For detail see: https://arxiv.org/pdf/1710.10044.pdf
    '''

    def __init__(self, env_info,
                 quantile_dist_function_builder=default_quantile_dist_function_builder,
                 params=QRDQNParam(),
                 replay_buffer_builder=default_replay_buffer_builder):
        super(QRDQN, self).__init__(env_info, params=params)

        if self._params.kappa == 0.0:
            logger.info(
                "kappa is set to 0.0. {} will use quantile regression loss".format(self.__name__))
        else:
            logger.info(
                "kappa is non 0.0. {} will use quantile huber loss".format(self.__name__))

        if not self._env_info.is_discrete_action_env():
            raise ValueError(
                '{} only supports discrete action environment'.format(self.__name__))
        state_shape = self._env_info.observation_space.shape
        self._n_action = self._env_info.action_space.n

        self._qj = 1 / self._params.num_quantiles

        self._quantile_dist = quantile_dist_function_builder(
            'quantile_dist_train', state_shape, self._n_action, self._params.num_quantiles)
        assert isinstance(self._quantile_dist, M.QuantileDistributionFunction)

        self._target_quantile_dist = quantile_dist_function_builder(
            'quantile_dist_target', state_shape, self._n_action, self._params.num_quantiles)
        assert isinstance(self._target_quantile_dist,
                          M.QuantileDistributionFunction)

        tau_hat = self._precompute_tau_hat(self._params.num_quantiles)
        self._tau_hat_var = nn.Variable.from_numpy_array(tau_hat)

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
        super(QRDQN, self)._post_init()

        copy_network_parameters(
            self._quantile_dist.get_parameters(),
            self._target_quantile_dist.get_parameters(),
            tau=1.0)

    def compute_eval_action(self, state):
        action, _ = epsilon_greedy_action_selection(state,
                                                    self._greedy_action_selector,
                                                    self._random_action_selector,
                                                    epsilon=self._params.test_epsilon)
        return action

    def _build_training_graph(self):
        target_quantiles = self._target_quantile_dist.quantiles(
            self._training_variables.s_next)
        a_star = self._compute_argmax_q(target_quantiles)

        theta_j = self._quantiles_of(target_quantiles, a_star)
        Ttheta_j = self._training_variables.reward + \
            self._training_variables.non_terminal * self._params.gamma * theta_j
        Ttheta_j = RF.expand_dims(Ttheta_j, axis=1)
        Ttheta_j.need_grad = False
        assert Ttheta_j.shape == (
            self._params.batch_size, 1, self._params.num_quantiles)

        Ttheta_i = self._quantile_dist.quantiles(
            s=self._training_variables.s_current)
        Ttheta_i = self._quantiles_of(Ttheta_i,
                                      self._training_variables.a_current)
        Ttheta_i = RF.expand_dims(Ttheta_i, axis=2)
        assert Ttheta_i.shape == (self._params.batch_size,
                                  self._params.num_quantiles,
                                  1)

        tau_hat = RF.expand_dims(self._tau_hat_var, axis=0)
        tau_hat = RF.repeat(tau_hat, repeats=self._params.batch_size, axis=0)
        tau_hat = RF.expand_dims(tau_hat, axis=2)
        assert tau_hat.shape == Ttheta_i.shape

        quantile_huber_loss = RF.quantile_huber_loss(
            Ttheta_j, Ttheta_i, self._params.kappa, tau_hat)
        assert quantile_huber_loss.shape == (self._params.batch_size,
                                             self._params.num_quantiles,
                                             self._params.num_quantiles)

        quantile_huber_loss = NF.mean(quantile_huber_loss, axis=2)
        quantile_huber_loss = NF.sum(quantile_huber_loss, axis=1)
        self._quantile_huber_loss = NF.mean(quantile_huber_loss)

    def _build_evaluation_graph(self):
        quantiles = self._quantile_dist.quantiles(
            self._evaluation_variables.s_eval)
        self._a_greedy = self._compute_argmax_q(quantiles)

    def _setup_solver(self):
        self._quantile_dist_solver = NS.Adam(
            alpha=self._params.learning_rate, eps=1e-2 / self._params.batch_size)
        self._quantile_dist_solver.set_parameters(
            self._quantile_dist.get_parameters())

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
            self._qrdqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._qrdqn_training(buffer)

    def _qrdqn_training(self, replay_buffer):
        if self.iteration_num % self._params.learner_update_frequency != 0:
            return

        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)

        self._training_variables.s_current.d = s
        self._training_variables.a_current.d = a
        self._training_variables.reward.d = r
        self._training_variables.non_terminal.d = non_terminal
        self._training_variables.s_next.d = s_next

        self._quantile_dist_solver.zero_grad()
        self._quantile_huber_loss.forward()
        self._quantile_huber_loss.backward()
        self._quantile_dist_solver.update()

        # Update target net
        if self.iteration_num % self._params.target_update_frequency == 0:
            copy_network_parameters(
                self._quantile_dist.get_parameters(),
                self._target_quantile_dist.get_parameters(),
                tau=1.0)

    def _greedy_action_selector(self, s):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._a_greedy.forward()
        return self._a_greedy.d

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, ))

    def _precompute_tau_hat(self, num_quantiles):
        tau_hat = [(tau_prev + tau_i) / num_quantiles / 2.0
                   for tau_prev, tau_i in zip(range(0, num_quantiles), range(1, num_quantiles+1))]
        return np.array(tau_hat, dtype=np.float32)

    def _compute_argmax_q(self, quantiles):
        q_values = self._compute_q_values(quantiles)
        return RF.argmax(q_values, axis=1)

    def _compute_q_values(self, quantiles):
        batch_size = quantiles.shape[0]
        assert quantiles.shape == (
            batch_size, self._n_action, self._params.num_quantiles)
        q_values = NF.sum(quantiles * self._qj, axis=2)
        assert q_values.shape == (batch_size, self._n_action)
        return q_values

    def _quantiles_of(self, quantiles, a):
        batch_size = quantiles.shape[0]
        quantiles = NF.transpose(quantiles, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        quantiles = quantiles * one_hot
        quantiles = NF.sum(quantiles, axis=2)
        assert quantiles.shape == (batch_size, self._params.num_quantiles)

        return quantiles

    def _to_one_hot(self, a):
        batch_size = a.shape[0]
        a = NF.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = NF.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=(
            batch_size, self._params.num_quantiles, self._n_action))
        return one_hot

    def _models(self):
        models = {}
        models[self._quantile_dist.scope_name] = self._quantile_dist
        models[self._target_quantile_dist.scope_name] = self._target_quantile_dist
        return models

    def _solvers(self):
        solvers = {}
        solvers['quantile_dist_solver'] = self._quantile_dist_solver
        return solvers
