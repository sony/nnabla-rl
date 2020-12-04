import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.models import REINFORCEContinousPolicy, REINFORCEDiscretePolicy, StochasticPolicy
import nnabla_rl.model_trainers as MT


def build_continuous_policy(scope_name, env_info, algorithm_params, **kwargs):
    return REINFORCEContinousPolicy(scope_name,
                                    env_info.state_dim,
                                    env_info.action_dim,
                                    algorithm_params.fixed_ln_var)


def build_discrete_policy(scope_name, env_info, algorithm_params, **kwargs):
    return REINFORCEDiscretePolicy(scope_name, env_info.state_dim, env_info.action_dim)


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
        self._assert_positive(self.num_rollouts_per_train_iteration, 'num_rollouts_per_train_iteration')
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
            if self._env_info.is_discrete_action_env():
                self._policy = build_discrete_policy("pi", self._env_info, self._params)
            else:
                self._policy = build_continuous_policy("pi", self._env_info, self._params)
        assert isinstance(self._policy, StochasticPolicy)

        def policy_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._policy_solver = {self._policy.scope_name: policy_solver_builder()}

        self._state = None
        self._action = None
        self._next_state = None
        self._buffer = None

    def compute_eval_action(self, s):
        return self._compute_action(s)

    def _before_training_start(self, env_or_buffer):
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.SPGPolicyTrainerParam(
            pi_loss_scalar=1.0/self._params.num_rollouts_per_train_iteration,
            grad_clip_norm=self._params.clip_grad_norm)
        policy_trainer = MT.policy_trainers.SPGPolicyTrainer(
            env_info=self._env_info,
            params=policy_trainer_params)

        training = MT.policy_trainings.REINFORCETraining()
        policy_trainer.setup_training(self._policy, self._policy_solver, training)
        return policy_trainer

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

    def _reinforce_training(self, buffer):
        # sample all experience in the buffer
        experiences, *_ = buffer.sample(buffer.capacity)
        s_batch, a_batch, accumulated_reward_batch = self._align_experiences_and_compute_accumulated_reward(experiences)

        self._policy_trainer.train((s_batch, a_batch, accumulated_reward_batch))

    def _align_experiences_and_compute_accumulated_reward(self, experiences):
        s_batch = None
        a_batch = None
        accumulated_reward_batch = None

        for experience in experiences:
            s_seq, a_seq, r_seq, _, _ = marshall_experiences(experience)
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

    def _compute_action(self, s):
        s_eval_var = nn.Variable.from_numpy_array(np.expand_dims(s, axis=0))
        with nn.auto_forward():
            distribution = self._policy.pi(s_eval_var)
            eval_action = distribution.sample()
        return eval_action.d.flatten()

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._policy_solver)
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        return latest_iteration_state
