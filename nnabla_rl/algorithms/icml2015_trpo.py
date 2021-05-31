# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import gym
import numpy as np

import nnabla as nn
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import ICML2015TRPOAtariPolicy, ICML2015TRPOMujocoPolicy, StochasticPolicy
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences


@dataclass
class ICML2015TRPOConfig(AlgorithmConfig):
    '''ICML2015TRPO config
    Args:
        gamma (float): Discount factor of rewards. Defaults to 0.99.
        num_steps_per_iteration (int): Number of steps per each training iteration for collecting on-policy experinces.\
            Increasing this step size is effective to get precise parameters of policy and value function updating, \
            but computational time of each iteration will increase. Defaults to 100000.
        batch_size (int): Trainig batch size of policy. \
            Usually, batch_size is the same as num_steps_per_iteration. Defaults to 100000.
        gpu_batch_size (int, optional): Actual batch size to reduce one forward gpu calculation memory. \
            As long as gpu memory size is enough, this configuration should not be specified. If not specified,  \
            gpu_batch_size is the same as pi_batch_size. Defaults to None.
        sigma_kl_divergence_constraint (float): Constraint size of kl divergence \
            between previous policy and updated policy. Defaults to 0.01.
        maximum_backtrack_numbers (int): Maximum backtrack numbers of linesearch. Defaults to 10.
        conjugate_gradient_damping (float): Damping size of conjugate gradient method. Defaults to 0.1.
        conjugate_gradient_iterations (int): Number of iterations of conjugate gradient method. Defaults to 20.
    '''
    gamma: float = 0.99
    num_steps_per_iteration: int = int(1e5)
    batch_size: int = int(1e5)
    gpu_batch_size: Optional[int] = None
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.001
    conjugate_gradient_iterations: int = 10

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.batch_size, 0, self.num_steps_per_iteration, 'batch_size')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: ICML2015TRPOConfig,
                    **kwargs) -> StochasticPolicy:
        if env_info.is_discrete_action_env():
            return self._build_default_discrete_policy(scope_name, env_info, algorithm_config)
        else:
            return self._build_default_continuous_policy(scope_name, env_info, algorithm_config)

    def _build_default_continuous_policy(self,
                                         scope_name: str,
                                         env_info: EnvironmentInfo,
                                         algorithm_config: ICML2015TRPOConfig,
                                         **kwargs) -> StochasticPolicy:
        return ICML2015TRPOMujocoPolicy(scope_name, env_info.action_dim)

    def _build_default_discrete_policy(self,
                                       scope_name: str,
                                       env_info: EnvironmentInfo,
                                       algorithm_config: ICML2015TRPOConfig,
                                       **kwargs) -> StochasticPolicy:
        return ICML2015TRPOAtariPolicy(scope_name, env_info.action_dim)


class ICML2015TRPO(Algorithm):
    '''Trust Region Policy Optimiation method with Single Path algorithm.

    This class implements the Trust Region Policy Optimiation (TRPO)
    with Single Path algorithm proposed by J. Schulman, et al. in the paper: "Trust Region Policy Optimization"
    For detail see: https://arxiv.org/abs/1502.05477

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`ICML2015TRPOConfig <nnabla_rl.algorithms.icml2015_trpo.ICML2015TRPOConfig>`):
            configuration of ICML2015TRPO algorithm
        policy_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ICML2015TRPOConfig
    _policy: StochasticPolicy
    _policy_trainer: ModelTrainer
    _environment_explorer: EnvironmentExplorer
    _eval_state_var: nn.Variable
    _eval_action: nn.Variable

    _policy_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: ICML2015TRPOConfig = ICML2015TRPOConfig(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder()):
        super(ICML2015TRPO, self).__init__(env_or_env_info, config=config)
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._policy = policy_builder("pi", self._env_info, self._config)

    @eval_api
    def compute_eval_action(self, s):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._compute_action(s)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_config = EE.RawPolicyExplorerConfig(initial_step_num=self.iteration_num, timelimit_as_terminal=False)
        explorer = EE.RawPolicyExplorer(
            policy_action_selector=self._compute_action, env_info=self._env_info, config=explorer_config)
        return explorer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.TRPOPolicyTrainerConfig(
            gpu_batch_size=self._config.gpu_batch_size,
            sigma_kl_divergence_constraint=self._config.sigma_kl_divergence_constraint,
            maximum_backtrack_numbers=self._config.maximum_backtrack_numbers,
            conjugate_gradient_damping=self._config.conjugate_gradient_damping,
            conjugate_gradient_iterations=self._config.conjugate_gradient_iterations)
        policy_trainer = MT.policy_trainers.TRPOPolicyTrainer(
            model=self._policy,
            env_info=self._env_info,
            config=policy_trainer_config)

        return policy_trainer

    def _run_online_training_iteration(self, env):
        if self.iteration_num % self._config.num_steps_per_iteration != 0:
            return

        buffer = ReplayBuffer(capacity=self._config.num_steps_per_iteration)

        num_steps = 0
        while num_steps <= self._config.num_steps_per_iteration:
            experience = self._environment_explorer.rollout(env)
            buffer.append(experience)
            num_steps += len(experience)

        self._trpo_training(buffer)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _trpo_training(self, buffer):
        buffer_iterator = BufferIterator(buffer, 1, shuffle=False, repeat=False)
        s, a, accumulated_reward = self._align_experiences(buffer_iterator)

        extra = {}
        extra['advantage'] = accumulated_reward  # Use accumulated_reward as advantage
        batch = TrainingBatch(batch_size=self._config.batch_size,
                              s_current=s,
                              a_current=a,
                              extra=extra)

        self._policy_trainer_state = self._policy_trainer.train(batch)

    def _align_experiences(self, buffer_iterator):
        s_batch = []
        a_batch = []
        accumulated_reward_batch = []

        buffer_iterator.reset()
        for experiences in buffer_iterator:
            experience, *_ = experiences
            s_seq, a_seq, r_seq, *_ = marshal_experiences(experience[0])
            accumulated_reward = self._compute_accumulated_reward(r_seq.flatten(), self._config.gamma)
            s_batch.append(s_seq)
            a_batch.append(a_seq)
            accumulated_reward_batch.append(accumulated_reward)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        accumulated_reward_batch = np.concatenate(accumulated_reward_batch, axis=0)

        return s_batch[:self._config.num_steps_per_iteration], \
            a_batch[:self._config.num_steps_per_iteration], \
            accumulated_reward_batch[:self._config.num_steps_per_iteration]

    def _compute_accumulated_reward(self, reward_sequence, gamma):
        if not reward_sequence.ndim == 1:
            raise ValueError("Invalid reward_sequence dimension")
        episode_length = len(reward_sequence)
        gamma_seq = np.array(
            [gamma**i for i in range(episode_length)])

        left_justified_gamma_seqs = np.tril(
            np.tile(gamma_seq, (episode_length, 1)), k=0)[::-1]
        mask = left_justified_gamma_seqs != 0.

        gamma_seqs = np.zeros((episode_length, episode_length))
        gamma_seqs[np.triu_indices(episode_length)] = left_justified_gamma_seqs[mask]

        return np.sum(reward_sequence*gamma_seqs, axis=1, keepdims=True)

    @eval_api
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
        latest_iteration_state = super(ICML2015TRPO, self).latest_iteration_state
        return latest_iteration_state
