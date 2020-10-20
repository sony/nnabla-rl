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
from nnabla_rl.models.model import Model
import nnabla_rl.functions as RF


def default_q_function_builder(scope_name, state_dim, action_dim):
    return M.SACQFunction(scope_name, state_dim, action_dim)


def default_policy_builder(scope_name, state_dim, action_dim):
    return M.SACPolicy(scope_name, state_dim, action_dim)


@dataclass
class SACParam(AlgorithmParam):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    environment_steps: int = 1
    gradient_steps: int = 1
    target_entropy: float = None
    initial_temperature: float = None
    fix_temperature: bool = False
    batch_size: int = 256
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 < self.gradient_steps):
            raise ValueError('gradient steps must be greater than 0')
        if not (0 < self.environment_steps):
            raise ValueError('environment steps must be greater than 0')
        if (self.initial_temperature is not None):
            if (self.initial_temperature <= 0.0):
                raise ValueError('temperature must be greater than 0')
        if not (0 <= self.start_timesteps):
            raise ValueError('start_timesteps must not be negative')


class AdjustableTemperature(Model):
    def __init__(self, scope_name, initial_value=None):
        super(AdjustableTemperature, self).__init__(scope_name=scope_name)
        if initial_value:
            initial_value = np.log(initial_value)
        else:
            initial_value = np.random.normal()

        initializer = np.reshape(initial_value, newshape=(1, 1))
        with nn.parameter_scope(scope_name):
            self._log_temperature = \
                nn.parameter.get_parameter_or_create(
                    name='log_temperature', shape=(1, 1), initializer=initializer, )
        # Dummy call. Just for initializing the parameters
        self()

    def __call__(self):
        return NF.exp(self._log_temperature)


class SAC(Algorithm):
    '''Soft Actor-Critic (SAC) algorithm implementation.

    This class implements the extended version of Soft Actor Critic (SAC) algorithm
    proposed by T. Haarnoja, et al. in the paper: "Soft Actor-Critic Algorithms and Applications"
    For detail see: https://arxiv.org/pdf/1812.05905.pdf

    This algorithm is slightly differs from the implementation of Soft Actor-Critic algorithm presented
    also by T. Haarnoja, et al. in the following paper:  https://arxiv.org/pdf/1801.01290.pdf

    The temperature parameter is adjusted automatically instead of providing reward scalar as a
    hyper parameter.

    '''

    def __init__(self, env_info,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 params=SACParam()):
        super(SAC, self).__init__(env_info, params=params)

        state_dim = env_info.observation_space.shape[0]
        action_dim = env_info.action_space.shape[0]

        self._q1 = q_function_builder(
            scope_name="q1", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._q1, M.QFunction)
        self._q2 = q_function_builder(
            scope_name="q2", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._q2, M.QFunction)
        self._pi = policy_builder(
            scope_name="pi", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._pi, M.StochasticPolicy)

        if self._params.fix_temperature & (self._params.initial_temperature is None):
            raise ValueError(
                'please set the initial temperature for fixed temperature training')
        self._alpha = AdjustableTemperature(
            scope_name="temperature", initial_value=self._params.initial_temperature)

        self._target_q1 = q_function_builder(
            scope_name="target_q1", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._target_q1, M.QFunction)
        self._target_q2 = q_function_builder(
            scope_name="target_q2", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._target_q2, M.QFunction)

        if self._params.target_entropy is None:
            self._params.target_entropy = -action_dim

        self._state = None
        self._action = None
        self._next_state = None
        self._episode_timesteps = None
        self._replay_buffer = ReplayBuffer(capacity=params.replay_buffer_size)

        # training input/loss variables
        self._s_current_var = nn.Variable((params.batch_size, state_dim))
        self._a_current_var = nn.Variable((params.batch_size, action_dim))
        self._s_next_var = nn.Variable((params.batch_size, state_dim))
        self._reward_var = nn.Variable((params.batch_size, 1))
        self._non_terminal_var = nn.Variable((params.batch_size, 1))
        self._pi_loss = None
        self._q_loss = None
        self._alpha_loss = None

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, state_dim))
        self._eval_distribution = None

    def _post_init(self):
        super(SAC, self)._post_init()

        copy_network_parameters(
            self._q1.get_parameters(), self._target_q1.get_parameters(), 1.0)
        copy_network_parameters(
            self._q2.get_parameters(), self._target_q2.get_parameters(), 1.0)

    def compute_eval_action(self, state):
        return self._compute_greedy_action(state, deterministic=True)

    def _build_training_graph(self):
        # Critic optimization graph
        policy_distribution = self._pi.pi(self._s_next_var)
        sampled_action, log_pi = policy_distribution.sample_and_compute_log_prob()

        target_q1_var = self._target_q1.q(self._s_next_var, sampled_action)
        target_q2_var = self._target_q2.q(self._s_next_var, sampled_action)
        target_q_var = NF.minimum2(target_q1_var, target_q2_var)

        y = self._reward_var + self._params.gamma * \
            self._non_terminal_var * \
            (target_q_var - self._temperature * log_pi)
        y.need_grad = False

        current_q1 = self._q1.q(self._s_current_var, self._a_current_var)
        current_q2 = self._q2.q(self._s_current_var, self._a_current_var)

        q1_loss = 0.5 * RF.mean_squared_error(y, current_q1)
        q2_loss = 0.5 * RF.mean_squared_error(y, current_q2)
        self._q_loss = q1_loss + q2_loss

        # Actor optimization graph
        policy_distribution = self._pi.pi(self._s_current_var)
        action_var, log_pi = policy_distribution.sample_and_compute_log_prob()
        q1 = self._q1.q(self._s_current_var, action_var)
        q2 = self._q2.q(self._s_current_var, action_var)
        min_q = NF.minimum2(q1, q2)
        self._pi_loss = NF.mean(self._temperature * log_pi - min_q)

        log_pi_unlinked = log_pi.get_unlinked_variable()
        self._alpha_loss = -NF.mean(self._temperature *
                                    (log_pi_unlinked + self._params.target_entropy))

    def _build_evaluation_graph(self):
        self._eval_distribution = self._pi.pi(self._eval_state_var)

    def _setup_solver(self):
        self._q1_solver = NS.Adam(alpha=self._params.learning_rate)
        self._q1_solver.set_parameters(self._q1.get_parameters())

        self._q2_solver = NS.Adam(alpha=self._params.learning_rate)
        self._q2_solver.set_parameters(self._q2.get_parameters())

        self._pi_solver = NS.Adam(alpha=self._params.learning_rate)
        self._pi_solver.set_parameters(self._pi.get_parameters())

        self._alpha_solver = NS.Adam(alpha=self._params.learning_rate)
        self._alpha_solver.set_parameters(self._alpha.get_parameters())

    def _run_online_training_iteration(self, env):
        for _ in range(self._params.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._params.gradient_steps):
            self._run_gradient_step(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._sac_training(buffer)

    def _run_environment_step(self, env):
        if self._state is None:
            self._state = env.reset()
            self._episode_timesteps = 0
        self._episode_timesteps += 1

        if self.iteration_num < self._params.start_timesteps:
            self._action = env.action_space.sample()
        else:
            self._action = self._compute_greedy_action(self._state)

        self._next_state, r, done, _ = env.step(self._action)
        if done and self._episode_timesteps < self._env_info.max_episode_steps:
            non_terminal = 0.0
        else:
            non_terminal = 1.0
        experience = \
            (self._state, self._action, [r], [non_terminal], self._next_state)
        self._replay_buffer.append(experience)

        if done:
            self._state = env.reset()
            self._episode_timesteps = 0
        else:
            self._state = self._next_state

    def _run_gradient_step(self, replay_buffer):
        if self._params.start_timesteps < self.iteration_num:
            self._sac_training(replay_buffer)

    def _sac_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)
        # Optimize critic
        self._s_current_var.d = s
        self._a_current_var.d = a
        self._s_next_var.d = s_next
        self._reward_var.d = r
        self._non_terminal_var.d = non_terminal

        self._q_loss.forward()

        self._q1_solver.zero_grad()
        self._q2_solver.zero_grad()
        self._q_loss.backward()
        self._q1_solver.update()
        self._q2_solver.update()

        # Optimize actor
        self._pi_loss.forward()
        self._pi_solver.zero_grad()
        self._pi_loss.backward()
        self._pi_solver.update()

        # Update temperature if requested
        if not self._params.fix_temperature:
            self._alpha_loss.forward()
            self._alpha_solver.zero_grad()
            self._alpha_loss.backward()
            self._alpha_solver.update()

        copy_network_parameters(
            self._q1.get_parameters(), self._target_q1.get_parameters(), self._params.tau)
        copy_network_parameters(
            self._q2.get_parameters(), self._target_q2.get_parameters(), self._params.tau)

    def _compute_greedy_action(self, s, deterministic=False):
        self._eval_state_var.d = np.expand_dims(s, axis=0)
        if deterministic:
            eval_action = self._eval_distribution.choose_probable()
        else:
            eval_action = self._eval_distribution.sample()
        eval_action.forward(clear_buffer=True)
        return np.squeeze(eval_action.data.data, axis=0)

    @property
    def _temperature(self):
        return self._alpha()

    def _models(self):
        models = [self._q1, self._target_q1, self._q2,
                  self._target_q2, self._pi, self._alpha]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers['q1_solver'] = self._q1_solver
        solvers['q2_solver'] = self._q2_solver
        solvers['pi_solver'] = self._pi_solver
        if not self._params.fix_temperature:
            solvers['alpha_solver'] = self._alpha_solver
        return solvers
