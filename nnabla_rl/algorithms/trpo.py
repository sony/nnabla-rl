import nnabla as nn

import nnabla.functions as NF
import nnabla.solvers as NS

import numpy as np

from collections import namedtuple
from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.optimization import conjugate_gradient
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.algorithms.common_utils import compute_v_target_and_advantage
from nnabla_rl.logger import logger
import nnabla_rl.models as M
import nnabla_rl.functions as RF
import nnabla_rl.preprocessors as RP


@dataclass
class TRPOParam(AlgorithmParam):
    gamma: float = 0.995
    lmb: float = 0.97
    num_steps_per_iteration: int = 5000
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 20
    vf_epochs: int = 5
    vf_batch_size: int = 64
    vf_learning_rate: float = 1e-3

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.lmb, 0.0, 1.0, 'lmb')
        self._assert_positive(
            self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(
            self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(
            self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(
            self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(
            self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')
        self._assert_positive(self.vf_epochs, 'vf_epochs')
        self._assert_positive(self.vf_batch_size, 'vf_batch_size')
        self._assert_positive(self.vf_learning_rate, 'vf_learning_rate')


def build_mujoco_state_preprocessor(scope_name, state_shape):
    return RP.RunningMeanNormalizer(scope_name, state_shape, value_clip=(-5.0, 5.0))


def build_state_preprocessor(preprocessor_builder, state_shape, scope_name):
    if preprocessor_builder is None:
        return build_mujoco_state_preprocessor(scope_name, state_shape)
    else:
        return preprocessor_builder(scope_name, state_shape)


def build_default_policy(scope_name, state_dim, action_dim):
    return M.TRPOPolicy(scope_name, state_dim, action_dim)


def build_default_v_function(scope_name, state_dim):
    return M.TRPOVFunction(scope_name, state_dim)


def _hessian_vector_product(flat_grads, params, vector):
    """ Compute multiplied vector hessian of parameters and vector

    Args:
        flat_grads (nn.Variable): gradient of parameters, should be flattened
        params (list[nn.Variable]): list of parameters
        vector (numpy.ndarray): input vector, shape is the same as flat_grads
    Returns:
        hessian_vector (numpy.ndarray): multiplied vector of hessian of parameters and vector
    See:
        https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
    """
    assert flat_grads.shape[0] == len(vector)
    if isinstance(vector, np.ndarray):
        vector = nn.Variable.from_numpy_array(vector)
    hessian_multiplied_vector_loss = NF.sum(flat_grads * vector)
    hessian_multiplied_vector_loss.forward()
    for param in params:
        param.grad.zero()
    hessian_multiplied_vector_loss.backward()
    hessian_multiplied_vector = [param.g.copy().flatten()
                                 for param in params]
    return np.concatenate(hessian_multiplied_vector)


def _concat_network_params_in_ndarray(params):
    """ Concatenate network parameters in numpy.ndarray,
        this function returns copied parameters

    Args:
        params (OrderedDict): parameters
    Returns:
        flat_params (numpy.ndarray): flatten parameters in numpy.ndarray type
    """
    flat_params = []
    for param in params.values():
        flat_param = param.d.copy().flatten()
        flat_params.append(flat_param)
    return np.concatenate(flat_params)


def _update_network_params_by_flat_params(params, new_flat_params):
    """ Update Network parameters by hand

    Args:
        params (OrderedDict): parameteres
        new_flat_params (numpy.ndarray): flattened new parameters
    """
    if not isinstance(new_flat_params, np.ndarray):
        raise ValueError("Invalid new_flat_params")
    total_param_numbers = 0
    for param in params.values():
        param_shape = param.shape
        param_numbers = len(param.d.flatten())
        new_param = new_flat_params[total_param_numbers:total_param_numbers +
                                    param_numbers].reshape(param_shape)
        param.d = new_param
        total_param_numbers += param_numbers
    assert total_param_numbers == len(new_flat_params)


class TRPO(Algorithm):
    """ Trust Region Policy Optimiation method
        with Generalized Advantage Estimation (GAE)
        See: https://arxiv.org/pdf/1502.05477.pdf and
            https://arxiv.org/pdf/1506.02438.pdf
    """

    def __init__(self, env_info,
                 value_function_builder=build_default_v_function,
                 policy_builder=build_default_policy,
                 state_preprocessor_builder=None,
                 params=TRPOParam()):
        super(TRPO, self).__init__(env_info, params=params)

        state_shape = self._env_info.observation_space.shape

        if self._env_info.is_discrete_action_env():
            self._state_preprocessor = None
        else:
            self._state_preprocessor = build_state_preprocessor(
                state_preprocessor_builder, state_shape, "preprocessor")

        if self._env_info.is_discrete_action_env():
            action_dim = self._env_info.action_space.n
        else:
            action_dim = self._env_info.action_space.shape[0]

        if self._env_info.is_discrete_action_env():
            raise NotImplementedError
        else:
            self._policy = policy_builder(
                "pi", state_shape[0], action_dim)
            self._old_policy = policy_builder(
                "old_pi", state_shape[0], action_dim)
            self._v_function = value_function_builder(
                "v", state_dim=state_shape[0])
            self._policy.set_state_preprocessor(self._state_preprocessor)
            self._old_policy.set_state_preprocessor(self._state_preprocessor)
            self._v_function.set_state_preprocessor(self._state_preprocessor)

        assert isinstance(self._policy, M.Policy)
        assert isinstance(self._old_policy, M.Policy)
        assert isinstance(self._v_function, M.VFunction)
        assert isinstance(self._state_preprocessor, RP.Preprocessor)

        self._params = params

        self._state = None
        self._action = None
        self._next_state = None
        self._buffer = None
        self._training_variables = None
        self._evaluation_variables = None
        self._v_loss = None

        self._create_variables(
            state_shape, action_dim, self._params.num_steps_per_iteration, self._params.vf_batch_size)

    def _post_init(self):
        super(TRPO, self)._post_init()

        copy_network_parameters(
            self._policy.get_parameters(), self._old_policy.get_parameters())

    def compute_eval_action(self, s):
        return self._compute_action(s)

    def _build_training_graph(self):
        # value function learning
        _v = self._v_function.v(self._training_variables.vf_s_current)
        self._v_loss = RF.mean_squared_error(
            _v, self._training_variables.vf_target)

        # policy learning
        distribution = self._policy.pi(
            self._training_variables.policy_s_current)
        old_distribution = self._old_policy.pi(
            self._training_variables.policy_s_current)

        self._kl_divergence = NF.mean(
            old_distribution.kl_divergence(distribution))

        _kl_divergence_grads = nn.grad(
            [self._kl_divergence], self._policy.get_parameters().values())

        self._kl_divergence_flat_grads = NF.concatenate(
            *[grad.reshape((-1,)) for grad in _kl_divergence_grads])
        self._kl_divergence_flat_grads.need_grad = True

        log_prob = distribution.log_prob(
            self._training_variables.policy_a_current)
        old_log_prob = old_distribution.log_prob(
            self._training_variables.policy_a_current)

        prob_ratio = NF.exp(log_prob - old_log_prob)
        self._approximate_return = NF.mean(
            prob_ratio*self._training_variables.advantage)

        _approximate_return_grads = nn.grad(
            [self._approximate_return], self._policy.get_parameters().values())

        self._approximate_return_flat_grads = NF.concatenate(
            *[grad.reshape((-1,)) for grad in _approximate_return_grads])
        self._approximate_return_flat_grads.need_grad = True

    def _build_evaluation_graph(self):
        distribution = self._policy.pi(
            self._evaluation_variables.policy_s_eval)
        self._eval_action = distribution.sample()

    def _setup_solver(self):
        self._v_solver = NS.Adam(alpha=self._params.vf_learning_rate)
        self._v_solver.set_parameters(self._v_function.get_parameters())

    def _run_online_training_iteration(self, env):

        if self.iteration_num % self._params.num_steps_per_iteration != 0:
            return

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
            self._next_state, r, done, info = env.step(self._action)
            truncated = info.get('TimeLimit.truncated', False)
            if done and not truncated:
                non_terminal = 0.0
            else:
                non_terminal = 1.0

            experience.append((self._state, self._action,
                               r, non_terminal, self._next_state))
            self._state = self._next_state

        return experience

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _create_variables(self, state_shape, action_dim, policy_batch_size, vf_batch_size):
        # Training input/loss variables
        Variables = namedtuple('Variables',
                               ['policy_s_current', 'policy_a_current', 'advantage',
                                'vf_s_current', 'vf_target'])

        policy_s_current_var = nn.Variable((policy_batch_size, *state_shape))
        policy_a_current_var = nn.Variable((policy_batch_size, action_dim))
        advantage_var = nn.Variable((policy_batch_size, 1))
        vf_s_current_var = nn.Variable((vf_batch_size, *state_shape))
        vf_target_var = nn.Variable((vf_batch_size, 1))

        self._training_variables = Variables(
            policy_s_current_var, policy_a_current_var, advantage_var, vf_s_current_var, vf_target_var)
        self._approximate_return = None
        self._approximate_return_flat_grads = None
        self._kl_divergence = None
        self._kl_divergence_flat_grads = None
        self._v_loss = None

        # Evaluation input variables
        policy_s_eval_var = nn.Variable((1, *state_shape))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['policy_s_eval'])
        self._evaluation_variables = EvaluationVariables(policy_s_eval_var)

    def _trpo_training(self, buffer):
        # sample all experience in the buffer
        experiences, *_ = buffer.sample(len(buffer))
        s_batch, a_batch, v_target_batch, adv_batch = \
            self._align_experiences(experiences)

        if self._state_preprocessor is not None:
            self._state_preprocessor.update(s_batch)

        # v function training
        self._v_function_training(s_batch, v_target_batch)

        # policy training
        self._policy_training(s_batch, a_batch, adv_batch)

    def _align_experiences(self, experiences):
        v_target_batch, adv_batch = self._compute_v_target_and_advantage(
            experiences)

        s_batch, a_batch = self._align_state_and_action(experiences)

        return s_batch[:self._params.num_steps_per_iteration], \
            a_batch[:self._params.num_steps_per_iteration], \
            v_target_batch[:self._params.num_steps_per_iteration], \
            adv_batch[:self._params.num_steps_per_iteration]

    def _compute_v_target_and_advantage(self, experiences):
        v_target_batch = []
        adv_batch = []
        for experience in experiences:
            v_target, adv = compute_v_target_and_advantage(
                self._v_function, experience, gamma=self._params.gamma, lmb=self._params.lmb)
            v_target_batch.append(v_target.reshape(-1, 1))
            adv_batch.append(adv.reshape(-1, 1))

        adv_batch = np.concatenate(adv_batch, axis=0)
        v_target_batch = np.concatenate(v_target_batch, axis=0)

        adv_mean = np.mean(adv_batch)
        adv_std = np.std(adv_batch)
        adv_batch = (adv_batch - adv_mean) / adv_std
        return v_target_batch, adv_batch

    def _align_state_and_action(self, experiences):
        s_batch = []
        a_batch = []

        for experience in experiences:
            s_seq, a_seq, _, _, _ = marshall_experiences(
                experience)
            s_batch.append(s_seq)
            a_batch.append(a_seq)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        return s_batch, a_batch

    def _v_function_training(self, s_batch, v_target_batch):
        data_size = len(s_batch)
        num_iterations_per_epoch = data_size // self._params.vf_batch_size
        for _ in range(self._params.vf_epochs * num_iterations_per_epoch):
            idx = np.random.randint(
                0, data_size, size=self._params.vf_batch_size)
            self._training_variables.vf_s_current.d = s_batch[idx]
            self._training_variables.vf_target.d = v_target_batch[idx]

            self._v_loss.forward()
            self._v_solver.zero_grad()
            self._v_loss.backward()
            self._v_solver.update()

    def _policy_training(self, s_batch, a_batch, adv_batch):
        full_step_params_update = self._compute_full_step_params_update(
            s_batch, a_batch, adv_batch)

        self._linesearch_and_update_params(
            s_batch, a_batch, adv_batch, full_step_params_update)

        copy_network_parameters(self._policy.get_parameters(
        ), self._old_policy.get_parameters(), tau=1.0)

    def _params_zero_grad(self):
        for param in self._policy.get_parameters().values():
            param.grad.zero()

    def _compute_full_step_params_update(self, s_batch, a_batch, adv_batch):
        _, _, approximate_return_flat_grads = \
            self._forward_variables(s_batch, a_batch, adv_batch)

        step_direction = conjugate_gradient(
            self._fisher_vector_product, approximate_return_flat_grads,
            max_iterations=self._params.conjugate_gradient_iterations)

        sAs = float(np.dot(step_direction,
                           self._fisher_vector_product(step_direction)))

        beta = (2.0 * self._params.sigma_kl_divergence_constraint /
                (sAs + 1e-8)) ** 0.5  # adding 1e-8 to avoid computational error
        full_step_params_update = beta * step_direction

        return full_step_params_update

    def _fisher_vector_product(self, vector):
        self._params_zero_grad()
        self._kl_divergence_flat_grads.forward()

        hessian_multiplied_vector = \
            _hessian_vector_product(self._kl_divergence_flat_grads,
                                    self._policy.get_parameters().values(),
                                    vector) + self._params.conjugate_gradient_damping * vector

        return hessian_multiplied_vector

    def _linesearch_and_update_params(self, s_batch, a_batch, adv_batch,
                                      full_step_params_update):
        current_flat_params = _concat_network_params_in_ndarray(
            self._policy.get_parameters())

        current_approximate_return, _, _ = self._forward_variables(s_batch, a_batch,
                                                                   adv_batch)

        for step_size in 0.5**np.arange(self._params.maximum_backtrack_numbers):
            new_flat_params = current_flat_params + step_size * full_step_params_update
            _update_network_params_by_flat_params(
                self._policy.get_parameters(), new_flat_params)

            approximate_return, kl_divergence, _ = self._forward_variables(
                s_batch, a_batch, adv_batch)

            improved = approximate_return - current_approximate_return > 0.
            is_in_kl_divergence_constraint = kl_divergence < self._params.sigma_kl_divergence_constraint

            if improved and is_in_kl_divergence_constraint:
                return
            elif not improved:
                logger.debug(
                    "TRPO LineSearch: Not improved, Shrink step size and Retry")
            elif not is_in_kl_divergence_constraint:
                logger.debug(
                    "TRPO LineSearch: Not fullfill constraints, Shrink step size and Retry")
            else:
                raise RuntimeError("Should not reach here")

        logger.debug(
            "TRPO LineSearch: Reach max iteration so Recover current parmeteres...")
        _update_network_params_by_flat_params(
            self._policy.get_parameters(), current_flat_params)

    def _forward_variables(self, s_batch, a_batch, adv_batch):
        self._training_variables.policy_s_current.d = s_batch
        self._training_variables.policy_a_current.d = a_batch
        self._training_variables.advantage.d = adv_batch

        nn.forward_all([self._approximate_return,
                        self._approximate_return_flat_grads,
                        self._kl_divergence])

        return float(self._approximate_return.d), \
            float(self._kl_divergence.d), \
            self._approximate_return_flat_grads.d

    def _compute_action(self, s, return_log_prob=True):
        self._evaluation_variables.policy_s_eval.d = np.expand_dims(s, axis=0)
        self._eval_action.forward()
        return self._eval_action.d.flatten()

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._old_policy.scope_name] = self._old_policy
        models[self._v_function.scope_name] = self._v_function
        models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers["v_solver"] = self._v_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(TRPO, self).latest_iteration_state()
        return latest_iteration_state
