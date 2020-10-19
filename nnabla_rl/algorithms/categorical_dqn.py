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


def default_value_distribution_builder(scope_name, state_dim, action_dim, num_atoms):
    return M.C51ValueDistributionFunction(scope_name, state_dim, action_dim, num_atoms)


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


@dataclass
class CategoricalDQNParam(AlgorithmParam):
    batch_size: int = 32
    gamma: float = 0.99
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    max_explore_steps: int = 1000000
    learning_rate: float = 0.00025
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51


class CategoricalDQN(Algorithm):
    '''Categorical DQN algorithm implementation.

    This class implements the Categorical DQN algorithm
    proposed by M. Bellemare, et al. in the paper: "A Distributional Perspective on Reinfocement Learning"
    For detail see: https://arxiv.org/pdf/1707.06887.pdf
    '''

    def __init__(self, env_info,
                 value_distribution_builder=default_value_distribution_builder,
                 replay_buffer_builder=default_replay_buffer_builder,
                 params=CategoricalDQNParam()):
        super(CategoricalDQN, self).__init__(env_info, params=params)
        if not self._env_info.is_discrete_action_env():
            raise ValueError(
                '{} only supports discrete action environment'.format(self.__name__))
        state_shape = self._env_info.observation_space.shape
        self._n_action = self._env_info.action_space.n

        N = self._params.num_atoms
        self._delta_z = (self._params.v_max - self._params.v_min) / (N - 1)

        self._atom_p = value_distribution_builder(
            'atom_p_train', state_shape, self._n_action, self._params.num_atoms)
        assert isinstance(self._atom_p, M.ValueDistributionFunction)

        self._target_atom_p = value_distribution_builder(
            'atom_p_target', state_shape, self._n_action, self._params.num_atoms)
        assert isinstance(self._target_atom_p, M.ValueDistributionFunction)

        self._z = self._precompute_z(self._params.num_atoms,
                                     self._params.v_min,
                                     self._params.v_max)
        self._z_var = nn.Variable.from_numpy_array(self._z)

        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = replay_buffer_builder(params.replay_buffer_size)

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
        s_next_var = nn.Variable((params.batch_size, *state_shape))
        mi_var = nn.Variable((params.batch_size, params.num_atoms))

        TrainingVariables = namedtuple(
            'TrainingVariables', ['s_current', 'a_current',  's_next', 'mi'])
        self._training_variables = \
            TrainingVariables(s_current_var, a_current_var, s_next_var, mi_var)

        # Training loss/output
        self._pj = None
        self._cross_entropy_loss = None

        # Evaluation input variables
        s_eval_var = nn.Variable((1, *state_shape))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['s_eval'])
        self._evaluation_variables = EvaluationVariables(s_eval_var)

        # Evaluation output
        self._a_greedy = None

    def _post_init(self):
        super(CategoricalDQN, self)._post_init()
        copy_network_parameters(
            self._atom_p.get_parameters(),
            self._target_atom_p.get_parameters(), tau=1.0)

    def compute_eval_action(self, state):
        action, _ = epsilon_greedy_action_selection(state,
                                                    self._greedy_action_selector,
                                                    self._random_action_selector,
                                                    epsilon=self._params.test_epsilon)
        return action

    def _build_training_graph(self):
        target_atom_probabilities = self._target_atom_p.probabilities(
            self._training_variables.s_next)
        a_star = self._compute_argmax_q(target_atom_probabilities)
        self._pj = self._probabilities_of(target_atom_probabilities, a_star)
        self._pj.need_grad = False

        atom_probabilities = self._atom_p.probabilities(
            self._training_variables.s_current)
        atom_probabilities = self._probabilities_of(
            atom_probabilities,
            self._training_variables.a_current)
        atom_probabilities = F.clip_by_value(atom_probabilities, 1e-10, 1.0)
        cross_entropy = self._training_variables.mi * F.log(atom_probabilities)
        assert cross_entropy.shape == (
            self._params.batch_size, self._params.num_atoms)
        self._cross_entropy_loss = -F.mean(F.sum(cross_entropy, axis=1))

    def _build_evaluation_graph(self):
        atom_probabilities = self._atom_p.probabilities(
            self._evaluation_variables.s_eval)
        self._a_greedy = self._compute_argmax_q(atom_probabilities)

    def _setup_solver(self):
        self._atom_p_solver = S.Adam(
            alpha=self._params.learning_rate, eps=1e-2 / self._params.batch_size)
        self._atom_p_solver.set_parameters(self._atom_p.get_parameters())

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
            self._categorical_dqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._categorical_dqn_training(buffer)

    def _categorical_dqn_training(self, replay_buffer):
        if self.iteration_num % self._params.learner_update_frequency != 0:
            return

        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)

        self._training_variables.s_current.d = s
        self._training_variables.a_current.d = a
        self._training_variables.s_next.d = s_next

        self._pj.forward()

        z = np.broadcast_to(array=self._z,
                            shape=(self._params.batch_size, self._params.num_atoms))
        Tz = np.clip(r + non_terminal * self._params.gamma * z,
                     self._params.v_min,
                     self._params.v_max)
        assert Tz.shape == (self._params.batch_size, self._params.num_atoms)

        mi = self._compute_projection(Tz, self._pj.d)
        self._training_variables.mi.d = mi

        self._atom_p_solver.zero_grad()
        self._cross_entropy_loss.forward()
        self._cross_entropy_loss.backward()
        self._atom_p_solver.update()

        # Update target net
        if self.iteration_num % self._params.target_update_frequency == 0:
            copy_network_parameters(
                self._atom_p.get_parameters(),
                self._target_atom_p.get_parameters(), tau=1.0)

    def _compute_projection(self, Tz, pj):
        bj = (Tz - self._params.v_min) / self._delta_z
        bj = np.clip(bj, 0, self._params.num_atoms - 1)

        lower = np.floor(bj)
        upper = np.ceil(bj)
        assert lower.shape == (self._params.batch_size, self._params.num_atoms)
        assert upper.shape == (self._params.batch_size, self._params.num_atoms)

        offset = np.arange(0,
                           self._params.batch_size * self._params.num_atoms,
                           self._params.num_atoms,
                           dtype=np.int32)[..., None]
        ml_indices = (lower + offset).astype(np.int32)
        mu_indices = (upper + offset).astype(np.int32)

        mi = np.zeros(shape=(self._params.batch_size, self._params.num_atoms),
                      dtype=np.float32)
        # Fix upper - bj = bj - lower = 0 (Prevent not getting both 0. upper - l must always be 1)
        # upper - bj = (1 + lower) - bj
        upper = 1 + lower
        np.add.at(mi.ravel(),
                  ml_indices.ravel(),
                  (pj * (upper - bj)).ravel())
        np.add.at(mi.ravel(),
                  mu_indices.ravel(),
                  (pj * (bj - lower)).ravel())

        return mi

    def _greedy_action_selector(self, s):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._a_greedy.forward()
        return self._a_greedy.d

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, ))

    def _compute_argmax_q(self, atom_probabilities):
        q_values = self._compute_q_values(atom_probabilities)
        return RF.argmax(q_values, axis=1)

    def _compute_q_values(self, atom_probabilities):
        batch_size = atom_probabilities.shape[0]
        assert atom_probabilities.shape == (
            batch_size, self._n_action, self._params.num_atoms)
        z = RF.expand_dims(self._z_var, axis=0)
        z = RF.expand_dims(z, axis=1)
        z = F.broadcast(
            z, shape=(batch_size, self._n_action, self._params.num_atoms))
        q_values = F.sum(z * atom_probabilities, axis=2)
        assert q_values.shape == (batch_size, self._n_action)
        return q_values

    def _probabilities_of(self, probabilities, a):
        batch_size = probabilities.shape[0]
        probabilities = F.transpose(probabilities, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        probabilities = probabilities * one_hot
        probabilities = F.sum(probabilities, axis=2)
        assert probabilities.shape == (batch_size, self._params.num_atoms)

        return probabilities

    def _to_one_hot(self, a):
        batch_size = a.shape[0]
        a = F.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = F.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = F.broadcast(one_hot, shape=(
            batch_size, self._params.num_atoms, self._n_action))
        return one_hot

    def _precompute_z(self, num_atoms, v_min, v_max):
        delta_z = (v_max - v_min) / (num_atoms - 1)
        z = [v_min + i * delta_z for i in range(num_atoms)]
        return np.asarray(z)

    def _models(self):
        models = {}
        models[self._atom_p.scope_name] = self._atom_p
        return models

    def _solvers(self):
        solvers = {}
        solvers['atom_p_solver'] = self._atom_p_solver
        return solvers
