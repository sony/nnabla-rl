import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.models import REINFORCEContinousPolicy, REINFORCEDiscretePolicy, StochasticPolicy
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def build_continuous_policy(scope_name, env_info, algorithm_params, **kwargs):
    return REINFORCEContinousPolicy(scope_name,
                                    env_info.state_dim,
                                    env_info.action_dim,
                                    algorithm_params.fixed_ln_var)


def build_discrete_policy(scope_name, env_info, algorithm_params, **kwargs):
    return REINFORCEDiscretePolicy(scope_name, env_info.state_dim, env_info.action_dim)


def default_policy_builder(scope_name, env_info, algorithm_params, **kwargs):
    if env_info.is_discrete_action_env():
        policy = build_discrete_policy("pi", env_info, algorithm_params)
    else:
        policy = build_continuous_policy("pi", env_info, algorithm_params)
    return policy


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
                 policy_builder=default_policy_builder,
                 params=REINFORCEParam()):
        super(REINFORCE, self).__init__(env_info, params=params)

        self._policy = policy_builder("pi", self._env_info, self._params)
        assert isinstance(self._policy, StochasticPolicy)

        def policy_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._policy_solver = {self._policy.scope_name: policy_solver_builder()}

    @eval_api
    def compute_eval_action(self, s):
        action, _ = self._compute_action(s)
        return action

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_params = EE.RawPolicyExplorerParam(
            reward_scalar=self._params.reward_scale,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                        env_info=self._env_info,
                                        params=explorer_params)
        return explorer

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
        buffer = ReplayBuffer(capacity=self._params.num_rollouts_per_train_iteration)

        for _ in range(self._params.num_rollouts_per_train_iteration):
            experience = self._environment_explorer.rollout(env)
            buffer.append(experience)

        self._reinforce_training(buffer)

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
            s_seq, a_seq, r_seq, *_ = marshall_experiences(experience)
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
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._policy.pi(self._eval_state_var)
            self._eval_action = distribution.sample()
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

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
