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


def default_critic_builder(scope_name, env_info):
    state_dim = env_info.observation_space.shape[0]
    action_dim = env_info.action_space.shape[0]
    return M.TD3QFunction(scope_name, state_dim, action_dim)


def default_actor_builder(scope_name, env_info):
    state_dim = env_info.observation_space.shape[0]
    action_dim = env_info.action_space.shape[0]
    max_action_value = float(env_info.action_space.high[0])
    return M.TD3Policy(scope_name, state_dim, action_dim, max_action_value=max_action_value)


@dataclass
class DDPGParam(AlgorithmParam):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000


class DDPG(Algorithm):
    def __init__(self, env_info,
                 critic_builder=default_critic_builder,
                 actor_builder=default_actor_builder,
                 params=DDPGParam()):
        super(DDPG, self).__init__(env_info, params=params)

        self._q = critic_builder(scope_name="q", env_info=env_info)
        assert isinstance(self._q, M.Model)
        self._pi = actor_builder(scope_name="pi", env_info=env_info)
        assert isinstance(self._pi, M.Model)

        self._target_q = critic_builder(
            scope_name="target_q", env_info=env_info)
        assert isinstance(self._q, M.Model)
        self._target_pi = actor_builder(
            scope_name="target_pi", env_info=env_info)
        assert isinstance(self._target_pi, M.Model)

        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = ReplayBuffer(capacity=params.replay_buffer_size)
        self._episode_timesteps = None

        # training input/loss variables
        state_dim = env_info.observation_space.shape[0]
        action_dim = env_info.action_space.shape[0]

        self._s_current_var = nn.Variable((params.batch_size, state_dim))
        self._a_current_var = nn.Variable((params.batch_size, action_dim))
        self._s_next_var = nn.Variable((params.batch_size, state_dim))
        self._reward_var = nn.Variable((params.batch_size, 1))
        self._non_terminal_var = nn.Variable((params.batch_size, 1))
        self._pi_loss = None
        self._critic_loss = None

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, state_dim))
        self._eval_action = None

    def _post_init(self):
        super(DDPG, self)._post_init()

        copy_network_parameters(
            self._q.get_parameters(), self._target_q.get_parameters(), tau=1.0)
        copy_network_parameters(
            self._pi.get_parameters(), self._target_pi.get_parameters(), tau=1.0)

    def compute_eval_action(self, state):
        return self._compute_greedy_action(state)

    def _build_training_graph(self):
        # Critic optimization graph
        a_next_var = self._target_pi.pi(self._s_next_var)
        target_q_var = self._target_q.q(self._s_next_var, a_next_var)

        y = self._reward_var + self._params.gamma * \
            self._non_terminal_var * target_q_var
        y.need_grad = False

        current_q = self._q.q(self._s_current_var, self._a_current_var)

        self._critic_loss = RF.mean_squared_error(y, current_q)

        # Actor optimization graph
        action_var = self._pi.pi(self._s_current_var)
        q = self._q.q(self._s_current_var, action_var)
        self._pi_loss = -NF.mean(q)

    def _build_evaluation_graph(self):
        self._eval_action = self._pi.pi(self._eval_state_var)

    def _setup_solver(self):
        self._q_solver = NS.Adam(alpha=self._params.learning_rate)
        self._q_solver.set_parameters(self._q.get_parameters())

        self._pi_solver = NS.Adam(alpha=self._params.learning_rate)
        self._pi_solver.set_parameters(self._pi.get_parameters())

    def _run_online_training_iteration(self, env):
        if self._state is None:
            self._state = env.reset()
            self._episode_timesteps = 0
        self._episode_timesteps += 1

        if self.iteration_num < self._params.start_timesteps:
            self._action = env.action_space.sample()
        else:
            self._action = self._compute_greedy_action(self._state)
            self._action = self._append_noise(
                self._action, env.action_space.low, env.action_space.high)

        self._next_state, r, done, _ = env.step(self._action)

        if done and self._episode_timesteps < self._env_info.max_episode_steps:
            non_terminal = 0.0
        else:
            non_terminal = 1.0
        experience = \
            (self._state, self._action, [r], [non_terminal], self._next_state)
        self._replay_buffer.append(experience)

        self._state = self._next_state

        if self._params.start_timesteps < self.iteration_num:
            self._ddpg_training(self._replay_buffer)

        if done:
            self._state = env.reset()
            self._episode_timesteps = 0

    def _run_offline_training_iteration(self, buffer):
        self._ddpg_training(buffer)

    def _ddpg_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next) = marshall_experiences(experiences)
        # Optimize critic
        self._s_current_var.d = s
        self._a_current_var.d = a
        self._s_next_var.d = s_next
        self._reward_var.d = r
        self._non_terminal_var.d = non_terminal

        self._critic_loss.forward()

        self._q_solver.zero_grad()
        self._critic_loss.backward()
        self._q_solver.update()

        # Optimize actor
        self._pi_loss.forward()
        self._pi_solver.zero_grad()
        self._pi_loss.backward()
        self._pi_solver.update()

        copy_network_parameters(
            self._q.get_parameters(), self._target_q.get_parameters(), tau=self._params.tau)
        copy_network_parameters(
            self._pi.get_parameters(), self._target_pi.get_parameters(), tau=self._params.tau)

    def _compute_greedy_action(self, s):
        self._eval_state_var.d = np.expand_dims(s, axis=0)
        self._eval_action.forward(clear_buffer=True)
        return np.squeeze((self._eval_action.data).data, axis=0)

    def _append_noise(self, action, low, high):
        noise = np.random.normal(
            loc=0.0, scale=0.1, size=action.shape).astype(np.float32)
        return np.clip(action + noise, low, high)

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        models[self._target_q.scope_name] = self._target_q
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        return models

    def _solvers(self):
        solvers = {}
        solvers['q_solver'] = self._q_solver
        solvers['pi_solver'] = self._pi_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['iteration'] = self._iteration_num
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}

        latest_iteration_state['scalar']['critic_loss'] = float(
            self._critic_loss.d.flatten())
        latest_iteration_state['scalar']['pi_loss'] = float(
            self._pi_loss.d.flatten())

        latest_iteration_state['histogram'].update(self._q.get_parameters())

        return latest_iteration_state
