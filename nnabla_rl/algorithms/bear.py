import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
import nnabla_rl.models as M
import nnabla_rl.functions as RF
from nnabla_rl.models.model import Model
from nnabla_rl.distributions import Gaussian


def default_q_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return M.TD3QFunction(scope_name, env_info.state_dim, env_info.action_dim)


def default_policy_builder(scope_name, env_info, algorithm_params, **kwargs):
    return M.BEARPolicy(scope_name, env_info.state_dim, env_info.action_dim)


def default_vae_builder(scope_name, env_info, algorithm_params, **kwargs):
    return M.UnsquashedVariationalAutoEncoder(scope_name,
                                              env_info.state_dim,
                                              env_info.action_dim,
                                              env_info.action_dim*2)


@dataclass
class BEARParam(AlgorithmParam):
    '''BEARParam
    Parameters used in BEAR algorithm.

    Args:
        tau(float): soft network parameter update coefficient. Defaults to 0.005.
        gamma(float): reward decay. Defaults to 0.99.
        learning_rate(float): learning rate which is set for solvers. Defaults to 1.0*1e-3.
        lmb(float): weight used for balancing the ratio of minQ and maxQ during q update. Defaults to 0.75.
        epsilon(float): inequality constraint constant used during dual gradient descent. Defaults to 0.05.
        num_q_ensembles(int): number of q ensembles . Defaults to 2.
        num_mmd_actions(int): number of actions to sample for computing maximum mean discrepancy (MMD). Defaults to 5.
        num_action_sampoles(int): number of actions to sample for computing target q values. Defaults to 10.
        mmd_type(str): kernel type used for MMD computation. laplacian or gaussian is supported. Defaults to gaussian.
        mmd_sigma(float): parameter used for adjusting the  MMD. Defaults to 20.0.
        use_stddev(bool): Use standard deviation. Defaults to False.
        stddev_coeff(float): Weight parameter for standard deviation. Defaults to 1.2.
                             It does not take affect if use_stddev is False.
        warmup_iterations(int): Number of iterations until start updating the policy. Defaults to 20000
        start_timesteps(int or None): Number of iterations to start training the networks.
                                      Only used on online training and must be set on online training.
                                      Defaults to None.
        batch_size(int or None): Number of iterations starting to train the networks. Defaults to None.
        use_mean_for_eval(bool): Use mean value instead of best action among the samples for evaluation
    '''
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1e-3
    lmb: float = 0.75
    epsilon: float = 0.05
    num_q_ensembles: int = 2
    num_mmd_actions: int = 5
    num_action_samples: int = 10
    mmd_type: str = 'gaussian'
    mmd_sigma: float = 20.0
    use_stddev: bool = False
    stddev_coeff: float = 1.2
    fix_lagrange_multiplier: bool = False
    warmup_iterations: int = 20000
    start_timesteps: int = None
    batch_size: int = 100
    use_mean_for_eval: bool = False

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 <= self.num_q_ensembles):
            raise ValueError('num q ensembles must not be negative')
        if not (0 <= self.num_mmd_actions):
            raise ValueError('num mmd actions must not be negative')
        if not (0 <= self.num_action_samples):
            raise ValueError('num action samples must not be negative')
        if not (0 <= self.warmup_iterations):
            raise ValueError('warmup iterations must not be negative')
        if self.start_timesteps is not None:
            if not (0 <= self.start_timesteps):
                raise ValueError('start timesteps must not be negative')
        if not (0 <= self.batch_size):
            raise ValueError('batch size must not be negative')


class AdjustableLagrangeMultiplier(Model):
    def __init__(self, scope_name, initial_value=None):
        super(AdjustableLagrangeMultiplier, self).__init__(
            scope_name=scope_name)
        if initial_value:
            initial_value = np.log(initial_value)
        else:
            initial_value = np.random.normal()

        initializer = np.reshape(initial_value, newshape=(1, 1))
        with nn.parameter_scope(scope_name):
            self._log_lagrange = \
                nn.parameter.get_parameter_or_create(
                    name='log_lagrange', shape=(1, 1), initializer=initializer)
        # Dummy call. Just for initializing the parameters
        self()

    def __call__(self):
        return NF.exp(self._log_lagrange)

    def clip(self, min_value, max_value):
        self._log_lagrange.d = np.clip(
            self._log_lagrange.d, min_value, max_value)

    @property
    def value(self):
        return np.exp(self._log_lagrange.d)


class BEAR(Algorithm):
    '''Bootstrapping Error Accumulation Reduction (BEAR) algorithm implementation.

    This class implements the Bootstrapping Error Accumulation Reduction (BEAR) algorithm
    proposed by A. Kumar, et al. in the paper: "Stabilizing Off-Policy Q-learning via Bootstrapping Error Reduction"
    For detail see: https://arxiv.org/pdf/1906.00949.pdf

    '''

    def __init__(self, env_or_env_info,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 vae_builder=default_vae_builder,
                 params=BEARParam()):
        super(BEAR, self).__init__(env_or_env_info, params=params)

        self._q_ensembles = []
        self._target_q_ensembles = []
        for i in range(self._params.num_q_ensembles):
            q = q_function_builder(
                scope_name="q{}".format(i), env_info=self._env_info, algorithm_params=self._params)
            assert isinstance(q, M.QFunction)
            target_q = q_function_builder(
                scope_name="target_q{}".format(i), env_info=self._env_info, algorithm_params=self._params)
            self._q_ensembles.append(q)
            self._target_q_ensembles.append(target_q)

        self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._pi, M.StochasticPolicy)
        self._target_pi = policy_builder(scope_name="target_pi", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._target_pi, M.StochasticPolicy)

        self._vae = vae_builder(scope_name="vae", env_info=self._env_info, algorithm_params=self._params)
        self._lagrange = AdjustableLagrangeMultiplier(scope_name="alpha")

        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = ReplayBuffer(capacity=None)

        # training input/loss variables
        self._s_current_var = nn.Variable((params.batch_size, self._env_info.state_dim))
        self._a_current_var = nn.Variable((params.batch_size, self._env_info.action_dim))
        self._s_next_var = nn.Variable((params.batch_size, self._env_info.state_dim))
        self._reward_var = nn.Variable((params.batch_size, 1))
        self._non_terminal_var = nn.Variable((params.batch_size, 1))
        self._pi_warmup_loss = None
        self._pi_loss = None
        self._vae_loss = None
        self._q_loss = None
        self._y = None
        self._lagrange_loss = None

        latent_shape = (self._params.batch_size, self._env_info.action_dim * 2)
        self._target_latent_distribution = Gaussian(mean=np.zeros(shape=latent_shape, dtype=np.float32),
                                                    ln_var=np.zeros(shape=latent_shape, dtype=np.float32))

        # exploration input/action variables
        self._exploration_state_var = nn.Variable((1, self._env_info.state_dim))
        self._exploration_action = None

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, self._env_info.state_dim))
        self._eval_action = None
        self._eval_max_index = None

    def _post_init(self):
        super(BEAR, self)._post_init()
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            copy_network_parameters(
                q.get_parameters(), target_q.get_parameters(), 1.0)
        copy_network_parameters(self._pi.get_parameters(),
                                self._target_pi.get_parameters(),
                                1.0)

    def compute_eval_action(self, state):
        if self._params.use_mean_for_eval:
            self._eval_state_var.d = np.expand_dims(state, axis=0)
            self._eval_action.forward(clear_buffer=True)
            return np.squeeze(self._eval_action.d, axis=0)
        else:
            self._eval_state_var.d = np.expand_dims(state, axis=0)
            nn.forward_all([self._eval_action, self._eval_max_index])
            action = self._eval_action.d[self._eval_max_index.d[0]]
            return action

    def _build_training_graph(self):
        self._build_q_update_graph()
        self._build_vae_update_graph()
        self._build_policy_update_graph()
        self._build_exploration_graph()

    def _build_evaluation_graph(self):
        if self._params.use_mean_for_eval:
            eval_distribution = self._pi.pi(self._eval_state_var)
            self._eval_action = NF.tanh(eval_distribution.mean())
        else:
            repeat_num = 10
            state = RF.repeat(x=self._eval_state_var,
                              repeats=repeat_num, axis=0)
            assert state.shape == (repeat_num, self._eval_state_var.shape[1])
            eval_distribution = self._pi.pi(state)
            self._eval_action = NF.tanh(eval_distribution.sample())
            q_values = self._q_ensembles[0].q(state, self._eval_action)
            self._eval_max_index = RF.argmax(q_values, axis=0)

    def _setup_solver(self):
        self._q_solvers = []
        for q in self._q_ensembles:
            solver = NS.Adam(alpha=self._params.learning_rate)
            solver.set_parameters(q.get_parameters())
            self._q_solvers.append(solver)

        self._pi_solver = NS.Adam(alpha=self._params.learning_rate)
        self._pi_solver.set_parameters(self._pi.get_parameters())

        self._vae_solver = NS.Adam(alpha=self._params.learning_rate)
        self._vae_solver.set_parameters(self._vae.get_parameters())

        self._lagrange_solver = NS.Adam(alpha=self._params.learning_rate)
        self._lagrange_solver.set_parameters(self._lagrange.get_parameters())

    def _run_online_training_iteration(self, env):
        if self._params.start_timesteps is None:
            raise ValueError('Start timesteps must be set')
        if self._state is None:
            self._state = env.reset()

        if self.iteration_num < self._params.start_timesteps:
            self._action = env.action_space.sample()
        else:
            self._action = self._compute_exploration_action(self._state)

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
            self._bear_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._bear_training(buffer)

    def _build_q_update_graph(self):
        s_next_rep = RF.repeat(
            x=self._s_next_var, repeats=self._params.num_action_samples, axis=0)
        policy_distribution = self._target_pi.pi(s_next_rep)
        self._pi_ln_var = policy_distribution._ln_var
        a_next_rep = NF.tanh(policy_distribution.sample())
        q_values = NF.stack(*(q_target.q(s_next_rep, a_next_rep)
                              for q_target in self._target_q_ensembles))
        assert q_values.shape == (self._params.num_q_ensembles,
                                  self._params.batch_size * self._params.num_action_samples,
                                  1)
        weighted_q_minmax = self._params.lmb * NF.min(q_values, axis=0) \
            + (1.0 - self._params.lmb) * NF.max(q_values, axis=0)
        assert weighted_q_minmax.shape == (
            self._params.batch_size * self._params.num_action_samples, 1)

        next_q_value = NF.max(
            NF.reshape(weighted_q_minmax, shape=(self._params.batch_size, -1)), axis=1, keepdims=True)
        assert next_q_value.shape == (self._params.batch_size, 1)
        target_q_value = self._reward_var + self._params.gamma * \
            self._non_terminal_var * next_q_value
        target_q_value.need_grad = False

        loss = 0.0
        for q in self._q_ensembles:
            loss += RF.mean_squared_error(target_q_value,
                                          q.q(self._s_current_var, self._a_current_var))
        self._q_loss = loss

    def _build_policy_update_graph(self):
        unsquashed_sampled_actions = self._vae.decode_multiple(
            self._params.num_mmd_actions, self._s_current_var)
        policy_distribution = self._pi.pi(self._s_current_var)
        unsquashed_pi_actions = policy_distribution.sample_multiple(
            num_samples=self._params.num_mmd_actions, noise_clip=(-0.5, 0.5))
        squashed_pi_actions = NF.tanh(unsquashed_pi_actions)

        if self._params.mmd_type == 'gaussian':
            mmd_loss = self._compute_gaussian_mmd(
                unsquashed_sampled_actions, unsquashed_pi_actions, sigma=self._params.mmd_sigma)
        elif self._params.mmd_type == 'laplacian':
            mmd_loss = self._compute_laplacian_mmd(
                unsquashed_sampled_actions, unsquashed_pi_actions, sigma=self._params.mmd_sigma)
        else:
            raise ValueError(
                'Unknown mmd type: {}'.format(self._params.mmd_type))
        assert mmd_loss.shape == (self._params.batch_size, 1)

        s_hat = RF.expand_dims(self._s_current_var, axis=0)
        s_hat = RF.repeat(s_hat, repeats=self._params.num_mmd_actions, axis=0)
        s_hat = NF.reshape(s_hat, shape=(self._params.batch_size *
                                         self._params.num_mmd_actions, self._s_current_var.shape[-1]))
        a_hat = NF.transpose(squashed_pi_actions, axes=(1, 0, 2))
        a_hat = NF.reshape(a_hat, shape=(self._params.batch_size *
                                         self._params.num_mmd_actions,
                                         self._a_current_var.shape[-1]))

        q_values = NF.stack(*(q.q(s_hat, a_hat) for q in self._q_ensembles))
        assert q_values.shape == (
            self._params.num_q_ensembles, self._params.num_mmd_actions * self._params.batch_size, 1)
        q_values = NF.reshape(q_values, shape=(
            self._params.num_q_ensembles, self._params.num_mmd_actions, self._params.batch_size, 1))
        # Compute mean among sampled actions
        q_values = NF.mean(q_values, axis=1)
        assert q_values.shape == (
            self._params.num_q_ensembles, self._params.batch_size, 1)

        # Compute the minimum among ensembles
        q_min = NF.min(q_values, axis=0)

        assert q_min.shape == (self._params.batch_size, 1)
        # Compute stddev among q funciton ensembles
        if self._params.use_stddev:
            q_stddev = RF.std(x=q_values, axis=0, keepdims=False)
            assert q_stddev.shape == (self._params.batch_size, 1)
        else:
            q_stddev = 0.0

        self._pi_loss = NF.mean(-q_min +
                                q_stddev * self._params.stddev_coeff +
                                self._lagrange() * mmd_loss)
        self._pi_warmup_loss = NF.mean(self._lagrange() * mmd_loss)

        # Must forward pi_loss before forwarding lagrange_loss
        self._lagrange_loss = -NF.mean(-q_min +
                                       q_stddev * self._params.stddev_coeff +
                                       self._lagrange() * (mmd_loss - self._params.epsilon))

    def _build_vae_update_graph(self):
        latent_distribution, reconstructed_action = self._vae(
            self._s_current_var, self._a_current_var)
        squashed_action = NF.tanh(reconstructed_action)
        reconstruction_loss = RF.mean_squared_error(
            self._a_current_var, squashed_action)
        kl_divergence = \
            latent_distribution.kl_divergence(self._target_latent_distribution)
        latent_loss = 0.5 * NF.mean(kl_divergence)
        self._vae_loss = reconstruction_loss + latent_loss

    def _build_exploration_graph(self):
        exploration_distribution = self._pi.pi(self._exploration_state_var)
        self._exploration_action = NF.tanh(exploration_distribution.sample())

    def _bear_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)
        # Optimize critic
        self._s_current_var.d = s
        self._a_current_var.d = a
        self._s_next_var.d = s_next
        self._reward_var.d = r
        self._non_terminal_var.d = non_terminal

        # Train q functions
        self._q_loss.forward()
        for q_solver in self._q_solvers:
            q_solver.zero_grad()
        self._q_loss.backward()
        for q_solver in self._q_solvers:
            q_solver.update()

        # Train vae
        self._vae_loss.forward()
        self._vae_solver.zero_grad()
        self._vae_loss.backward()
        self._vae_solver.update()

        # Optimize actor
        # Always forward pi loss to update the graph
        if self.iteration_num < self._params.warmup_iterations:
            nn.forward_all(
                [self._pi_warmup_loss, self._lagrange_loss])
            self._pi_solver.zero_grad()
            self._pi_warmup_loss.backward()
            self._pi_solver.update()
        else:
            nn.forward_all([self._pi_loss, self._lagrange_loss])
            self._pi_solver.zero_grad()
            self._pi_loss.backward()
            self._pi_solver.update()

        # Update lagrange_multiplier if requested
        if not self._params.fix_lagrange_multiplier:
            self._lagrange_solver.zero_grad()
            self._lagrange_loss.backward()
            self._lagrange_solver.update()

        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            copy_network_parameters(
                q.get_parameters(), target_q.get_parameters(), self._params.tau)
        copy_network_parameters(self._pi.get_parameters(),
                                self._target_pi.get_parameters(),
                                self._params.tau)
        self._lagrange.clip(-5.0, 10.0)

    def _compute_exploration_action(self, s):
        self._exploration_state_var.d = np.expand_dims(s, axis=0)
        self._exploration_action.forward(clear_buffer=True)
        return np.squeeze(self._exploration_action.d, axis=0)

    def _compute_gaussian_mmd(self, samples1, samples2, sigma):
        n = samples1.shape[1]
        m = samples2.shape[1]

        k_xx = RF.expand_dims(x=samples1, axis=2) - \
            RF.expand_dims(x=samples1, axis=1)
        last_axis = len(k_xx.shape) - 1
        sum_k_xx = NF.sum(
            NF.exp(-NF.sum(k_xx**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_xy = RF.expand_dims(x=samples1, axis=2) - \
            RF.expand_dims(x=samples2, axis=1)
        last_axis = len(k_xy.shape) - 1
        sum_k_xy = NF.sum(
            NF.exp(-NF.sum(k_xy**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_yy = RF.expand_dims(x=samples2, axis=2) - \
            RF.expand_dims(x=samples2, axis=1)
        last_axis = len(k_yy.shape) - 1
        sum_k_yy = NF.sum(
            NF.exp(-NF.sum(k_yy**2, axis=last_axis, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        mmd_squared = \
            (sum_k_xx / (n*n) - 2.0 * sum_k_xy / (m*n) + sum_k_yy / (m*m))
        # Add 1e-6 to avoid numerical instability
        return RF.sqrt(mmd_squared + 1e-6)

    def _compute_laplacian_mmd(self, samples1, samples2, sigma):
        n = samples1.shape[1]
        m = samples2.shape[1]

        k_xx = RF.expand_dims(x=samples1, axis=2) - \
            RF.expand_dims(x=samples1, axis=1)

        sum_k_xx = NF.sum(
            NF.exp(-NF.sum(NF.abs(k_xx), axis=3, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_xy = RF.expand_dims(x=samples1, axis=2) - \
            RF.expand_dims(x=samples2, axis=1)
        sum_k_xy = NF.sum(
            NF.exp(-NF.sum(NF.abs(k_xy), axis=3, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_yy = RF.expand_dims(x=samples2, axis=2) - \
            RF.expand_dims(x=samples2, axis=1)
        sum_k_yy = NF.sum(
            NF.exp(-NF.sum(NF.abs(k_yy), axis=3, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        mmd_squared = \
            (sum_k_xx / (n*n) - 2.0 * sum_k_xy / (m*n) + sum_k_yy / (m*m))
        # Add 1e-6 to avoid numerical instability
        return RF.sqrt(mmd_squared + 1e-6)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._pi, self._target_pi,
                  self._vae, self._lagrange]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        for i, solver in enumerate(self._q_solvers):
            solvers['q{}_solver'.format(i)] = solver
        solvers['pi_solver'] = self._pi_solver
        solvers['vae_solver'] = self._vae_solver
        solvers['lagrange_solver'] = self._lagrange_solver
        return solvers

    @property
    def latest_iteration_state(self):
        state = super(BEAR, self).latest_iteration_state
        state['scalar']['vae_loss'] = self._vae_loss.d
        state['scalar']['pi_loss'] = self._pi_loss.d
        state['scalar']['q_loss'] = self._q_loss.d
        state['scalar']['lagrange_loss'] = self._lagrange_loss.d
        state['scalar']['lagrange_multiplier'] = self._lagrange.value
        state['scalar']['pi_stddev'] = np.mean(np.exp(self._pi_ln_var.d * 0.5))

        return state
