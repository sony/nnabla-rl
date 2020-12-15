import nnabla as nn

import numpy as np

from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.models import ICML2015TRPOAtariPolicy, ICML2015TRPOMujocoPolicy, StochasticPolicy
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


@dataclass
class ICML2015TRPOParam(AlgorithmParam):
    gamma: float = 0.99
    num_steps_per_iteration: int = int(1e5)
    batch_size: int = 2500
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.001
    conjugate_gradient_iterations: int = 10

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')


def build_default_continuous_policy(scope_name, env_info, algorithm_params, **kwargs):
    return ICML2015TRPOMujocoPolicy(scope_name, env_info.state_dim, env_info.action_dim)


def build_default_discrete_policy(scope_name, env_info, algorithm_params, **kwargs):
    return ICML2015TRPOAtariPolicy(scope_name, env_info.state_shape, env_info.action_dim)


class ICML2015TRPO(Algorithm):
    """ Trust Region Policy Optimiation method, this implements pure one.
        Please note that original TRPO use Single Path method to estimate Q value
        instead of Generalized Advantage Estimation (GAE).
        See: https://arxiv.org/pdf/1502.05477.pdf
    """

    def __init__(self, env_or_env_info, params=ICML2015TRPOParam()):
        super(ICML2015TRPO, self).__init__(env_or_env_info, params=params)

        if self._env_info.is_discrete_action_env():
            self._policy = build_default_discrete_policy("pi", self._env_info, self._params)
        else:
            self._policy = build_default_continuous_policy("pi", self._env_info, self._params)

        assert isinstance(self._policy, StochasticPolicy)

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
        explorer_params = EE.RawPolicyExplorerParam(initial_step_num=self.iteration_num, timelimit_as_terminal=False)
        explorer = EE.RawPolicyExplorer(
            policy_action_selector=self._compute_action, env_info=self._env_info, params=explorer_params)
        return explorer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.TRPOPolicyTrainerParam(
            batch_size=self._params.batch_size,
            num_steps_per_iteration=self._params.num_steps_per_iteration,
            sigma_kl_divergence_constraint=self._params.sigma_kl_divergence_constraint,
            maximum_backtrack_numbers=self._params.maximum_backtrack_numbers,
            conjugate_gradient_damping=self._params.conjugate_gradient_damping,
            conjugate_gradient_iterations=self._params.conjugate_gradient_iterations)
        policy_trainer = MT.policy_trainers.TRPOPolicyTrainer(
            env_info=self._env_info,
            params=policy_trainer_params)

        training = MT.model_trainer.Training()
        policy_trainer.setup_training(self._policy, {}, training)
        return policy_trainer

    def _run_online_training_iteration(self, env):
        self._buffer = ReplayBuffer(capacity=self._params.num_steps_per_iteration)

        num_steps = 0
        while num_steps <= self._params.num_steps_per_iteration:
            experience = self._environment_explorer.rollout(env)
            self._buffer.append(experience)
            num_steps += len(experience)

        self._trpo_training(self._buffer)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _trpo_training(self, buffer):
        # sample all experience in the buffer
        experiences, *_ = buffer.sample(len(buffer))
        s_batch, a_batch, advantage = self._align_experiences(experiences)
        extra = {}
        extra['advantage'] = advantage
        batch = TrainingBatch(batch_size=self._params.batch_size,
                              s_current=s_batch,
                              a_current=a_batch,
                              extra=extra)

        self._policy_trainer.train(batch)

    def _align_experiences(self, experiences):
        s_batch = []
        a_batch = []
        accumulated_reward_batch = []

        for experience in experiences:
            s_seq, a_seq, r_seq, *_ = marshall_experiences(experience)
            accumulated_reward = self._compute_accumulated_reward(r_seq, self._params.gamma)
            s_batch.append(s_seq)
            a_batch.append(a_seq)
            accumulated_reward_batch.append(accumulated_reward)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        accumulated_reward_batch = np.concatenate(accumulated_reward_batch, axis=0)

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
        gamma_seqs[np.triu_indices(episode_length)] = left_justified_gamma_seqs[mask]

        return np.sum(reward_sequence*gamma_seqs, axis=1, keepdims=True)

    def _compute_action(self, s):
        # evaluation input/action variables
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
        return {}

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        return latest_iteration_state
