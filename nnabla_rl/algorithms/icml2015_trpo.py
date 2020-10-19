import nnabla as nn

import nnabla.functions as F

import numpy as np

from collections import namedtuple
from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.optimization import conjugate_gradient
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.algorithms.trpo import _hessian_vector_product,\
    _concat_network_params_in_ndarray, _update_network_params_by_flat_params
from nnabla_rl.logger import logger
import nnabla_rl.models as M


@dataclass
class ICML2015TRPOParam(AlgorithmParam):
    gamma: float = 0.99
    num_steps_per_iteration: int = int(1e5)
    batch_size: int = 2500
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.001

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(
            self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(
            self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(
            self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(
            self.conjugate_gradient_damping, 'conjugate_gradient_damping')


def build_default_continuous_policy(scope_name, state_dim, action_dim):
    return M.ICML2015TRPOMujocoPolicy(scope_name, state_dim, action_dim)


def build_default_discrete_policy(scope_name, state_shape, action_dim):
    return M.ICML2015TRPOAtariPolicy(scope_name, state_shape, action_dim)


class ICML2015TRPO(Algorithm):
    """ Trust Region Policy Optimiation method, this implements pure one.
        Please note that original TRPO use Single Path method to estimate Q value
        instead of Generalized Advantage Estimation (GAE).
        See: https://arxiv.org/pdf/1502.05477.pdf
    """

    def __init__(self, env_info, params=ICML2015TRPOParam()):
        super(ICML2015TRPO, self).__init__(env_info, params=params)

        state_shape = self._env_info.observation_space.shape
        if self._env_info.is_discrete_action_env():
            action_dim = self._env_info.action_space.n
            self._policy = build_default_discrete_policy(
                "pi", state_shape, action_dim)
            self._old_policy = build_default_discrete_policy(
                "old_pi", state_shape, action_dim)
        else:
            action_dim = self._env_info.action_space.shape[0]
            self._policy = build_default_continuous_policy(
                "pi", state_shape[0], action_dim)
            self._old_policy = build_default_continuous_policy(
                "old_pi", state_shape[0], action_dim)

        assert isinstance(self._policy, M.Policy)
        self._params = params

        self._state = None
        self._action = None
        self._next_state = None
        self._buffer = None
        self._training_variables = None
        self._evaluation_variables = None

        self._create_variables(
            state_shape, action_dim, batch_size=self._params.batch_size)
        self._build_computation_graph()

        copy_network_parameters(
            self._policy.get_parameters(), self._old_policy.get_parameters())

    def compute_eval_action(self, s):
        return self._compute_action(s)

    def _build_training_graph(self):
        distribution = self._policy.pi(self._training_variables.s_current)

        old_distribution = self._old_policy.pi(
            self._training_variables.s_current)

        self._kl_divergence = F.mean(
            old_distribution.kl_divergence(distribution))

        _kl_divergence_grads = nn.grad(
            [self._kl_divergence], self._policy.get_parameters().values())

        self._kl_divergence_flat_grads = F.concatenate(
            *[grad.reshape((-1,)) for grad in _kl_divergence_grads])
        self._kl_divergence_flat_grads.need_grad = True

        log_prob = distribution.log_prob(self._training_variables.a_current)
        old_log_prob = old_distribution.log_prob(
            self._training_variables.a_current)

        prob_ratio = F.exp(log_prob - old_log_prob)
        self._approximate_return = F.mean(
            prob_ratio*self._training_variables.accumulated_reward)

        _approximate_return_grads = nn.grad(
            [self._approximate_return], self._policy.get_parameters().values())

        self._approximate_return_flat_grads = F.concatenate(
            *[grad.reshape((-1,)) for grad in _approximate_return_grads])
        self._approximate_return_flat_grads.need_grad = True

    def _build_evaluation_graph(self):
        distribution = self._policy.pi(self._evaluation_variables.s_eval)
        self._eval_action = distribution.sample()

    def _setup_solver(self):
        pass

    def _run_online_training_iteration(self, env):
        self._buffer = ReplayBuffer(
            capacity=self._params.num_steps_per_iteration)

        num_steps = 0
        while num_steps <= self._params.num_steps_per_iteration:
            experience = self._run_one_episode(env)
            self._buffer.append(experience)
            num_steps += len(experience)

        self._trpo_training(self._buffer)

    def _run_one_episode(self, env):
        self._state = env.reset()
        done = False
        experience = []

        while not done:
            self._action = self._compute_action(self._state)
            self._next_state, r, done, _ = env.step(self._action)
            non_terminal = np.float32(0.0 if done else 1.0)

            experience.append((self._state, self._action,
                               r, non_terminal, self._next_state))
            self._state = self._next_state

        return experience

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _create_variables(self, state_shape, action_dim, batch_size):
        # Training input/loss variables
        Variables = namedtuple('Variables',
                               ['s_current', 'a_current', 'accumulated_reward'])

        s_current_var = nn.Variable((batch_size, *state_shape))
        accumulated_reward_var = nn.Variable((batch_size, 1))

        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((batch_size, 1))
        else:
            a_current_var = nn.Variable((batch_size, action_dim))

        self._training_variables = Variables(
            s_current_var, a_current_var, accumulated_reward_var)
        self._approximate_return = None
        self._approximate_return_flat_grads = None
        self._kl_divergence = None
        self._kl_divergence_flat_grads = None

        # Evaluation input variables
        s_eval_var = nn.Variable((1, *state_shape))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['s_eval'])
        self._evaluation_variables = EvaluationVariables(s_eval_var)

    def _trpo_training(self, buffer):
        # sample all experience in the buffer
        experiences, *_ = buffer.sample(len(buffer))
        s_batch, a_batch, accumulated_reward_batch = self._align_experiences(
            experiences)

        full_step_params_update = self._compute_full_step_params_update(
            s_batch, a_batch, accumulated_reward_batch)

        self._linesearch_and_update_params(
            s_batch, a_batch, accumulated_reward_batch, full_step_params_update)

        copy_network_parameters(self._policy.get_parameters(
        ), self._old_policy.get_parameters(), tau=1.0)

    def _params_zero_grad(self):
        for param in self._policy.get_parameters().values():
            param.grad.zero()

    def _align_experiences(self, experiences):
        s_batch = []
        a_batch = []
        accumulated_reward_batch = []

        for experience in experiences:
            s_seq, a_seq, r_seq, _, _ = marshall_experiences(
                experience)
            accumulated_reward = self._compute_accumulated_reward(
                r_seq, self._params.gamma)
            s_batch.append(s_seq)
            a_batch.append(a_seq)
            accumulated_reward_batch.append(accumulated_reward)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        accumulated_reward_batch = np.concatenate(
            accumulated_reward_batch, axis=0)

        assert len(s_batch) >= self._params.num_steps_per_iteration
        return s_batch[:self._params.num_steps_per_iteration], \
            a_batch[:self._params.num_steps_per_iteration], \
            accumulated_reward_batch[:self._params.num_steps_per_iteration]

    def _compute_accumulated_reward(self, reward_sequence, gamma):
        episode_length = len(reward_sequence)
        gamma_seq = np.array(
            [gamma**i for i in range(episode_length)])

        left_justified_gamma_seqs = np.tril(
            np.tile(gamma_seq, (episode_length, 1)), k=0)[::-1]
        mask = left_justified_gamma_seqs != 0.

        gamma_seqs = np.zeros((episode_length, episode_length))
        gamma_seqs[np.triu_indices(episode_length)
                   ] = left_justified_gamma_seqs[mask]

        return np.sum(reward_sequence*gamma_seqs, axis=1, keepdims=True)

    def _set_batch_to_variable(self, iteration_numbers, s_batch, a_batch,
                               accumulated_reward_batch):
        start_idx = iteration_numbers*self._params.batch_size
        end_idx = (iteration_numbers+1)*self._params.batch_size
        self._training_variables.s_current.d = s_batch[start_idx:end_idx]
        self._training_variables.a_current.d = a_batch[start_idx:end_idx]
        self._training_variables.accumulated_reward.d = accumulated_reward_batch[
            start_idx:end_idx]

    def _compute_full_step_params_update(self, s_batch, a_batch,
                                         accumulated_reward_batch):

        _, _, approximate_return_flat_grads = \
            self._forward_variables(s_batch, a_batch,
                                    accumulated_reward_batch)

        def fisher_vector_product(vector):
            return self._fisher_vector_product(vector, s_batch, a_batch,
                                               accumulated_reward_batch)

        step_direction = conjugate_gradient(
            fisher_vector_product, approximate_return_flat_grads)

        sHs = float(np.dot(step_direction,
                           fisher_vector_product(step_direction)))
        scale = (2.0 * self._params.sigma_kl_divergence_constraint /
                 (sHs + 1e-8)) ** 0.5  # adding 1e-8 to avoid computational error
        full_step_params_update = scale * step_direction

        return full_step_params_update

    def _fisher_vector_product(self, vector, s_batch, a_batch,
                               accumulated_reward_batch):
        total_iteration_numbers = \
            self._params.num_steps_per_iteration // self._params.batch_size
        sum_hessian_multiplied_vector = np.zeros_like(vector)

        for i in range(total_iteration_numbers):
            self._set_batch_to_variable(
                i, s_batch, a_batch, accumulated_reward_batch)
            self._params_zero_grad()
            nn.forward_all([self._approximate_return, self._approximate_return_flat_grads,
                            self._kl_divergence, self._kl_divergence_flat_grads])

            hessian_multiplied_vector = \
                _hessian_vector_product(self._kl_divergence_flat_grads,
                                        self._policy.get_parameters().values(),
                                        vector) + self._params.conjugate_gradient_damping * vector

            sum_hessian_multiplied_vector += hessian_multiplied_vector

        return sum_hessian_multiplied_vector / total_iteration_numbers

    def _linesearch_and_update_params(self, s_batch, a_batch,
                                      accumulated_reward_batch, full_step_params_update):
        current_flat_params = _concat_network_params_in_ndarray(
            self._policy.get_parameters())

        current_approximate_return, _, _ = self._forward_variables(s_batch, a_batch,
                                                                   accumulated_reward_batch)

        for step_size in 0.5**np.arange(self._params.maximum_backtrack_numbers):
            new_flat_params = current_flat_params + step_size * full_step_params_update
            _update_network_params_by_flat_params(
                self._policy.get_parameters(), new_flat_params)

            approximate_return, kl_divergence, _ = self._forward_variables(
                s_batch, a_batch, accumulated_reward_batch
            )

            is_improved = approximate_return - current_approximate_return > 0.
            is_in_kl_divergence_constraint = kl_divergence < self._params.sigma_kl_divergence_constraint

            if is_improved and is_in_kl_divergence_constraint:
                return
            elif not is_improved:
                logger.debug(
                    "ICML2015TRPO LineSearch: Not improved, Shrink step size and Retry")
            elif not is_in_kl_divergence_constraint:
                logger.debug(
                    "ICML2015TRPO LineSearch: Not fullfill constraints, Shrink step size and Retry")

        logger.debug(
            "ICML2015TRPO LineSearch: Reach max iteration so Recover current parmeteres...")
        _update_network_params_by_flat_params(
            self._policy.get_parameters(), current_flat_params)

    def _forward_variables(self, s_batch, a_batch, accumulated_reward_batch):
        total_iteration_numbers = \
            self._params.num_steps_per_iteration // self._params.batch_size
        sum_approximate_return = 0.
        sum_kl_divergence = 0.
        sum_approximate_return_flat_grad = \
            np.zeros(self._approximate_return_flat_grads.shape)

        for i in range(total_iteration_numbers):
            self._set_batch_to_variable(
                i, s_batch, a_batch, accumulated_reward_batch)

            nn.forward_all([self._approximate_return, self._approximate_return_flat_grads,
                            self._kl_divergence])

            sum_approximate_return += float(self._approximate_return.d)
            sum_kl_divergence += float(self._kl_divergence.d)
            sum_approximate_return_flat_grad += self._approximate_return_flat_grads.d

        return sum_approximate_return / total_iteration_numbers, \
            sum_kl_divergence / total_iteration_numbers, \
            sum_approximate_return_flat_grad / total_iteration_numbers

    def _compute_action(self, s, return_log_prob=True):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._eval_action.forward()
        return self._eval_action.d.flatten()

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._old_policy.scope_name] = self._old_policy
        return models

    def _solvers(self):
        solvers = {}
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        return latest_iteration_state
