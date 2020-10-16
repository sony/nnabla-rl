import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S

from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
import nnabla_rl.models as M
from nnabla_rl.models.model import Model


def build_continuous_policy(scope_name, state_dim, action_dim, fixed_ln_var):
    return M.REINFORCEContinousPolicy(scope_name, state_dim, action_dim, fixed_ln_var)


def build_discrete_policy(scope_name, state_dim, action_dim):
    return M.REINFORCEDiscretePolicy(scope_name, state_dim, action_dim)


@dataclass
class REINFORCEParam(AlgorithmParam):
    reward_scale: float = 0.01
    num_rollouts_per_train_iteration: int = 10
    learning_rate: float = 1e-3
    clip_grad_norm: float = 1.
    # this parameter does not use in discrete environment
    fixed_ln_var: float = np.log(0.1)

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        self._assert_positive(self.reward_scale, 'reward_scale')
        self._assert_positive(
            self.num_rollouts_per_train_iteration, 'num_rollouts_per_train_iteration')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.clip_grad_norm, 'clip_grad_norm')


class REINFORCE(Algorithm):
    def __init__(self, env_info,
                 policy_builder=None,
                 params=REINFORCEParam()):
        super(REINFORCE, self).__init__(env_info, params=params)

        if policy_builder is not None:
            self._policy = policy_builder()
        else:
            state_shape = self._env_info.observation_space.shape
            if self._env_info.is_discrete_action_env():
                action_dim = self._env_info.action_space.n
                self._policy = build_discrete_policy(
                    "pi", state_shape[0], action_dim)
            else:
                action_dim = self._env_info.action_space.shape[0]
                self._policy = build_continuous_policy(
                    "pi", state_shape[0], action_dim, params.fixed_ln_var)

        assert isinstance(self._policy, M.Policy)

        self._state = None
        self._action = None
        self._next_state = None
        self._buffer = None
        self._training_variables = None
        self._evaluation_variables = None

        # The graph will be rebuilt when training runs
        self._create_variables(state_shape[0], action_dim, batch_size=1)

    def compute_eval_action(self, s):
        return self._compute_action(s)

    def _rebuild_computation_graph(self):
        self._build_computation_graph()

    def _build_training_graph(self):
        distribution = self._policy.pi(self._training_variables.s_current)
        log_prob = distribution.log_prob(self._training_variables.a_current)
        self._policy_loss = F.sum(-log_prob.reshape((-1, )) * self._training_variables.accumulated_reward) / \
            self._params.num_rollouts_per_train_iteration

    def _build_evaluation_graph(self):
        distribution = self._policy.pi(self._evaluation_variables.s_eval)
        self._eval_action = distribution.sample()

    def _setup_solver(self):
        self._policy_solver = S.Adam(alpha=self._params.learning_rate)
        self._policy_solver.set_parameters(self._policy.get_parameters())

    def _run_online_training_iteration(self, env):
        self._buffer = ReplayBuffer(
            capacity=self._params.num_rollouts_per_train_iteration)

        for _ in range(self._params.num_rollouts_per_train_iteration):
            self._state = env.reset()
            done = False
            experience = []

            while not done:
                self._action = self._compute_action(self._state)
                self._next_state, r, done, _ = env.step(self._action)
                non_terminal = np.float32(0.0 if done else 1.0)
                r *= self._params.reward_scale

                experience.append((self._state, self._action,
                                   r, non_terminal, self._next_state))
                self._state = self._next_state

            self._buffer.append(experience)

        self._reinforce_training(self._buffer)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _create_variables(self, state_dim, action_dim, batch_size):
        # Training input/loss variables
        Variables = namedtuple('Variables',
                               ['s_current', 'a_current', 'accumulated_reward'])

        s_current_var = nn.Variable((batch_size, state_dim))
        accumulated_reward_var = nn.Variable((batch_size, ))

        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((batch_size, 1))
        else:
            a_current_var = nn.Variable((batch_size, action_dim))

        self._training_variables = Variables(
            s_current_var, a_current_var, accumulated_reward_var)
        self._policy_loss = None

        # Evaluation input variables
        s_eval_var = nn.Variable((1, state_dim))

        EvaluationVariables = \
            namedtuple('EvaluationVariables', ['s_eval'])
        self._evaluation_variables = EvaluationVariables(s_eval_var)

    def _reinforce_training(self, buffer):
        # sample all experience in the buffer
        experiences, *_ = buffer.sample(buffer.capacity)
        s_batch, a_batch, accumulated_reward_batch = self._align_experiences_and_compute_accumulated_reward(
            experiences)

        # rebuild computational graph to fit batch size
        self._create_variables(
            s_batch.shape[1], a_batch.shape[1], len(s_batch))
        self._rebuild_computation_graph()

        self._training_variables.s_current.d = s_batch
        self._training_variables.a_current.d = a_batch
        self._training_variables.accumulated_reward.d = accumulated_reward_batch

        self._policy_loss.forward(clear_no_need_grad=True)
        self._policy_solver.zero_grad()
        self._policy_loss.backward(clear_buffer=True)
        self._policy_solver.clip_grad_by_norm(self._params.clip_grad_norm)
        self._policy_solver.update()

    def _align_experiences_and_compute_accumulated_reward(self, experiences):
        s_batch = None
        a_batch = None
        accumulated_reward_batch = None

        for experience in experiences:
            s_seq, a_seq, r_seq, non_terminal_seq, s_next_seq = marshall_experiences(
                experience)
            accumulated_reward = np.cumsum(r_seq[::-1])[::-1]

            if s_batch is None:
                s_batch = s_seq
                a_batch = a_seq
                accumulated_reward_batch = accumulated_reward
                continue

            s_batch = np.concatenate((s_batch, s_seq), axis=0)
            a_batch = np.concatenate((a_batch, a_seq), axis=0)
            accumulated_reward_batch = np.concatenate(
                (accumulated_reward_batch, accumulated_reward))

        return s_batch, a_batch, accumulated_reward_batch

    def _compute_action(self, s, return_log_prob=True):
        self._evaluation_variables.s_eval.d = np.expand_dims(s, axis=0)
        self._eval_action.forward()
        return self._eval_action.d.flatten()

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        return models

    def _solvers(self):
        solvers = {}
        solvers['policy_solver'] = self._policy_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}

        latest_iteration_state['scalar']['loss'] = self._policy_loss.d.flatten(
        )
        return latest_iteration_state
