import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
import nnabla_rl.models as M
from nnabla_rl.models.model import Model
import nnabla_rl.functions as RF


def default_v_function_builder(scope_name, state_dim):
    return M.SACVFunction(scope_name, state_dim)


def default_q_function_builder(scope_name, state_dim, action_dim):
    return M.SACQFunction(scope_name, state_dim, action_dim)


def default_policy_builder(scope_name, state_dim, action_dim):
    return M.SACPolicy(scope_name, state_dim, action_dim)


@dataclass
class ICML2018SACParam(AlgorithmParam):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    environment_steps: int = 1
    gradient_steps: int = 1
    reward_scalar: float = 5.0
    batch_size: int = 256
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    target_update_interval: int = 1

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.        

        '''
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.gradient_steps, 'gradient_steps')
        self._assert_positive(self.environment_steps, 'environment_steps')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.target_update_interval,
                              'target_update_interval')


class ICML2018SAC(Algorithm):
    '''Soft Actor-Critic (SAC) algorithm implementation.

    This class implements the ICML2018 version of Soft Actor Critic (SAC) algorithm proposed by T. Haarnoja, et al. 
    in the paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    For detail see: https://arxiv.org/pdf/1801.01290.pdf

    This implementation slightly differs from the implementation of Soft Actor-Critic algorithm presented 
    also by T. Haarnoja, et al. in the following paper: https://arxiv.org/pdf/1812.05905.pdf

    You will need to scale the reward received from the environment properly to get the algorithm work.
    '''

    def __init__(self, env_info,
                 v_function_builder=default_v_function_builder,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 params=ICML2018SACParam()):
        super(ICML2018SAC, self).__init__(env_info, params=params)

        state_dim = env_info.observation_space.shape[0]
        action_dim = env_info.action_space.shape[0]

        self._v = v_function_builder(
            scope_name="v", state_dim=state_dim)
        assert isinstance(self._v, M.VFunction)
        self._q1 = q_function_builder(
            scope_name="q1", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._q1, M.QFunction)
        self._q2 = q_function_builder(
            scope_name="q2", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._q2, M.QFunction)
        self._pi = policy_builder(
            scope_name="pi", state_dim=state_dim, action_dim=action_dim)
        assert isinstance(self._pi, M.StochasticPolicy)

        self._target_v = v_function_builder(
            scope_name="target_v", state_dim=state_dim)

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
        self._v_loss = None
        self._q_loss = None
        self._pi_loss = None

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, state_dim))
        self._eval_distribution = None

    def _post_init(self):
        super(ICML2018SAC, self)._post_init()
        copy_network_parameters(self._v.get_parameters(),
                                self._target_v.get_parameters(), 1.0)

    def compute_eval_action(self, state):
        return self._compute_greedy_action(state, deterministic=True)

    def _build_training_graph(self):
        # Critic optimization graph
        policy_distribution = self._pi.pi(self._s_current_var)
        sampled_action, log_pi = policy_distribution.sample_and_compute_log_prob()

        target_q1 = self._q1.q(self._s_current_var, sampled_action)
        target_q2 = self._q2.q(self._s_current_var, sampled_action)
        target_q_var = F.minimum2(target_q1, target_q2)

        y = (target_q_var - log_pi)
        y.need_grad = False

        current_v = self._v.v(self._s_current_var)
        self._v_loss = 0.5 * RF.mean_squared_error(y, current_v)

        target_v = self._target_v.v(self._s_next_var)
        q_hat = self._reward_var + self._params.gamma * self._non_terminal_var * target_v
        q_hat.need_grad = False
        current_q1 = self._q1.q(self._s_current_var, self._a_current_var)
        current_q2 = self._q2.q(self._s_current_var, self._a_current_var)
        q1_loss = 0.5 * RF.mean_squared_error(q_hat, current_q1)
        q2_loss = 0.5 * RF.mean_squared_error(q_hat, current_q2)
        self._q_loss = q1_loss + q2_loss

        # Actor optimization graph
        policy_distribution = self._pi.pi(self._s_current_var)
        sampled_action, log_pi = policy_distribution.sample_and_compute_log_prob()
        q1 = self._q1.q(self._s_current_var, sampled_action)
        q2 = self._q2.q(self._s_current_var, sampled_action)
        q = F.minimum2(q1, q2)
        self._pi_loss = F.mean(log_pi - q)

    def _build_evaluation_graph(self):
        self._eval_distribution = self._pi.pi(self._eval_state_var)

    def _setup_solver(self):
        self._v_solver = S.Adam(alpha=self._params.learning_rate)
        self._v_solver.set_parameters(self._v.get_parameters())

        self._q1_solver = S.Adam(alpha=self._params.learning_rate)
        self._q1_solver.set_parameters(self._q1.get_parameters())

        self._q2_solver = S.Adam(alpha=self._params.learning_rate)
        self._q2_solver.set_parameters(self._q2.get_parameters())

        self._pi_solver = S.Adam(alpha=self._params.learning_rate)
        self._pi_solver.set_parameters(self._pi.get_parameters())

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
        r = np.float32(r * self._params.reward_scalar)
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

        self._v_loss.forward()
        self._v_solver.zero_grad()
        self._v_loss.backward()
        self._v_solver.update()

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

        if self.iteration_num % self._params.target_update_interval == 0:
            copy_network_parameters(self._v.get_parameters(),
                                    self._target_v.get_parameters(),
                                    self._params.tau)

    def _compute_greedy_action(self, s, deterministic=False):
        self._eval_state_var.d = np.expand_dims(s, axis=0)
        if deterministic:
            eval_action = self._eval_distribution.choose_probable()
        else:
            eval_action = self._eval_distribution.sample()
        eval_action.forward(clear_buffer=True)
        return np.squeeze(eval_action.data.data, axis=0)

    def _models(self):
        models = [self._v, self._target_v, self._q1, self._q2, self._pi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers['v_solver'] = self._v_solver
        solvers['q1_solver'] = self._q1_solver
        solvers['q2_solver'] = self._q2_solver
        solvers['pi_solver'] = self._pi_solver
        return solvers
