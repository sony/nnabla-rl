# Copyright 2021 Sony Corporation.
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

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
import nnabla_rl.preprocessors as RP
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import (_StatePreprocessedPolicy, _StatePreprocessedRewardFunction,
                                               _StatePreprocessedVFunction, compute_v_target_and_advantage)
from nnabla_rl.builders import ModelBuilder, PreprocessorBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import (GAILDiscriminator, GAILPolicy, GAILVFunction, Model, RewardFunction, StochasticPolicy,
                              VFunction)
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences


@dataclass
class GAILConfig(AlgorithmConfig):
    '''GAIL config
    Args:
        act_deterministic_in_eval (bool): Enable act deterministically at evalution. Defaults to True.
        discriminator_batch_size (bool): Trainig batch size of discriminator.\
            Usually, discriminator_batch_size is the same as pi_batch_size. Defaults to 50000.
        discriminator_learning_rate (float): Learning rate which is set to the solvers of dicriminator function. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        discriminator_update_frequency (int): Frequency (measured in the number of parameter update) \
            of discriminator update. Defaults to 1.
        adversary_entropy_coef (float): Coefficient of entropy loss in dicriminator training. Defaults to 0.001.
        policy_update_frequency (int): Frequency (measured in the number of parameter update) \
            of policy update. Defaults to 1.
        gamma (float): Discount factor of rewards. Defaults to 0.995.
        lmb (float): Scalar of lambda return's computation in GAE. Defaults to 0.97.\
            This configuration is related to bias and variance of estimated value. \
            If it is close to 0, estimated value is low-variance but biased.\
            If it is close to 1, estimated value is unbiased but high-variance.
        num_steps_per_iteration (int): Number of steps per each training iteration for collecting on-policy experinces.\
            Increasing this step size is effective to get precise parameters of policy and value function updating, \
            but computational time of each iteration will increase. Defaults to 50000.
        pi_batch_size (int): Trainig batch size of policy. \
            Usually, pi_batch_size is the same as num_steps_per_iteration. Defaults to 50000.
        sigma_kl_divergence_constraint (float): Constraint size of kl divergence \
            between previous policy and updated policy. Defaults to 0.01.
        maximum_backtrack_numbers (int): Maximum backtrack numbers of linesearch. Defaults to 10.
        conjugate_gradient_damping (float): Damping size of conjugate gradient method. Defaults to 0.1.
        conjugate_gradient_iterations (int): Number of iterations of conjugate gradient method. Defaults to 10.
        vf_epochs (int): Number of epochs in each iteration. Defaults to 5.
        vf_batch_size (int): Training batch size of value function. Defaults to 128.
        vf_learning_rate (float): Learning rate which is set to the solvers of value function. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        preprocess_state (bool): Enable preprocessing the states in the collected experiences \
            before feeding as training batch. Defaults to True.
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    preprocess_state: bool = True
    act_deterministic_in_eval: bool = True
    discriminator_batch_size: int = 50000
    discriminator_learning_rate: float = 0.01
    discriminator_update_frequency: int = 1
    adversary_entropy_coef: float = 0.001
    policy_update_frequency: int = 1
    gamma: float = 0.995
    lmb: float = 0.97
    pi_batch_size: int = 50000
    num_steps_per_iteration: int = 50000
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 10
    vf_epochs: int = 5
    vf_batch_size: int = 128
    vf_learning_rate: float = 1e-3

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.pi_batch_size, 0, self.num_steps_per_iteration, 'pi_batch_size')
        self._assert_positive(self.discriminator_learning_rate, "discriminator_learning_rate")
        self._assert_positive(self.discriminator_batch_size, "discriminator_batch_size")
        self._assert_positive(self.policy_update_frequency, "policy_update_frequency")
        self._assert_positive(self.discriminator_update_frequency, "discriminator_update_frequency")
        self._assert_positive(self.adversary_entropy_coef, "adversarial_entropy_coef")
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.lmb, 0.0, 1.0, 'lmb')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')
        self._assert_positive(self.vf_epochs, 'vf_epochs')
        self._assert_positive(self.vf_batch_size, 'vf_batch_size')
        self._assert_positive(self.vf_learning_rate, 'vf_learning_rate')


class DefaultPreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(self,  # type: ignore[override]
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_config: GAILConfig,
                           **kwargs) -> Preprocessor:
        return RP.RunningMeanNormalizer(scope_name, env_info.state_shape, value_clip=(-5.0, 5.0))


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: GAILConfig,
                    **kwargs) -> StochasticPolicy:
        return GAILPolicy(scope_name, env_info.action_dim)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: GAILConfig,
                    **kwargs) -> VFunction:
        return GAILVFunction(scope_name)


class DefaultRewardFunctionBuilder(ModelBuilder[RewardFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: GAILConfig,
                    **kwargs) -> RewardFunction:
        return GAILDiscriminator(scope_name)


class DefaultVFunctionSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: GAILConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.vf_learning_rate)


class DefaultRewardFunctionSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: GAILConfig,
                     **kwargs) -> nn.solver.Solver:
        assert isinstance(algorithm_config, GAILConfig)
        return NS.Adam(alpha=algorithm_config.discriminator_learning_rate)


class GAIL(Algorithm):
    '''Generative Adversarial Imitation Learning implementation.

    This class implements the Generative Adversarial Imitation Learning (GAIL) algorithm
    proposed by Jonathan Ho, et al. in the paper: "Generative Adversarial Imitation Learning"
    For detail see: https://arxiv.org/abs/1606.03476

    This algorithm only supports online training.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        expert_buffer (:py:class:`ReplayBuffer <nnabla_rl.replay_buffer.ReplayBuffer>`):
            replay buffer which contains expert experience.
        config (:py:class:`GAILConfig <nnabla_rl.algorithms.gail.GAILConfig>`): configuration of GAIL algorithm
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for v function solvers
        policy_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
        reward_function_builder (:py:class:`ModelBuilder[RewardFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of reward function models
        reward_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for reward function solvers
        state_preprocessor_builder (None or :py:class:`PreprocessorBuilder <nnabla_rl.builders.PreprocessorBuilder>`):
            state preprocessor builder to preprocess the states
    '''

    _config: GAILConfig
    _v_function: VFunction
    _v_function_solver: nn.solver.Solver
    _policy: StochasticPolicy
    _discriminator: RewardFunction
    _discriminator_solver: nn.solver.Solver
    _environment_explorer: EnvironmentExplorer
    _v_function_trainer: ModelTrainer
    _policy_trainer: ModelTrainer
    _discriminator_trainer: ModelTrainer

    _s_var_label: nn.Variable
    _s_next_var_label: nn.Variable
    _a_var_label: nn.Variable
    _reward: nn.Variable

    _v_function_trainer_state: Dict[str, Any]
    _policy_trainer_state: Dict[str, Any]
    _discriminator_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 expert_buffer: ReplayBuffer,
                 config: GAILConfig = GAILConfig(),
                 v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultVFunctionSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 reward_function_builder: ModelBuilder[RewardFunction] = DefaultRewardFunctionBuilder(),
                 reward_solver_builder: SolverBuilder = DefaultRewardFunctionSolverBuilder(),
                 state_preprocessor_builder: Optional[PreprocessorBuilder] = DefaultPreprocessorBuilder()):
        super(GAIL, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_discrete_action_env():
            raise NotImplementedError

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            policy = policy_builder("pi", self._env_info, self._config)
            v_function = v_function_builder("v", self._env_info, self._config)
            discriminator = reward_function_builder("discriminator", self._env_info, self._config)

            if self._config.preprocess_state:
                if state_preprocessor_builder is None:
                    raise ValueError('State preprocessing is enabled but no preprocessor builder is given')
                pi_v_preprocessor = state_preprocessor_builder('pi_v_preprocessor', self._env_info, self._config)
                v_function = _StatePreprocessedVFunction(v_function=v_function, preprocessor=pi_v_preprocessor)
                policy = _StatePreprocessedPolicy(policy=policy, preprocessor=pi_v_preprocessor)
                r_preprocessor = state_preprocessor_builder('r_preprocessor', self._env_info, self._config)
                discriminator = _StatePreprocessedRewardFunction(
                    reward_function=discriminator, preprocessor=r_preprocessor)
                self._pi_v_state_preprocessor = pi_v_preprocessor
                self._r_state_preprocessor = r_preprocessor
            self._v_function = v_function
            self._policy = policy
            self._discriminator = discriminator

            self._v_function_solver = v_solver_builder(self._env_info, self._config)
            self._discriminator_solver = reward_solver_builder(self._env_info, self._config)

            self._expert_buffer = expert_buffer

    @eval_api
    def compute_eval_action(self, s):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._compute_action(s, act_deterministic=self._config.act_deterministic_in_eval)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)
        self._discriminator_trainer = self._setup_reward_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_config = EE.RawPolicyExplorerConfig(
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                        env_info=self._env_info,
                                        config=explorer_config)
        return explorer

    def _setup_v_function_training(self, env_or_buffer):
        v_function_trainer_config = MT.v_value_trainers.MonteCarloVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=1.0
        )
        v_function_trainer = MT.v_value_trainers.MonteCarloVTrainer(
            train_functions=self._v_function,
            solvers={self._v_function.scope_name: self._v_function_solver},
            env_info=self._env_info,
            config=v_function_trainer_config)
        return v_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.TRPOPolicyTrainerConfig(
            sigma_kl_divergence_constraint=self._config.sigma_kl_divergence_constraint,
            maximum_backtrack_numbers=self._config.maximum_backtrack_numbers,
            conjugate_gradient_damping=self._config.conjugate_gradient_damping,
            conjugate_gradient_iterations=self._config.conjugate_gradient_iterations)
        policy_trainer = MT.policy_trainers.TRPOPolicyTrainer(
            model=self._policy,
            env_info=self._env_info,
            config=policy_trainer_config)
        return policy_trainer

    def _setup_reward_function_training(self, env_or_buffer):
        reward_function_trainer_config = MT.reward_trainiers.GAILRewardFunctionTrainerConfig(
            batch_size=self._config.discriminator_batch_size,
            learning_rate=self._config.discriminator_learning_rate,
            entropy_coef=self._config.adversary_entropy_coef
        )
        reward_function_trainer = MT.reward_trainiers.GAILRewardFunctionTrainer(
            models=self._discriminator,
            solvers={self._discriminator.scope_name: self._discriminator_solver},
            env_info=self._env_info,
            config=reward_function_trainer_config)

        return reward_function_trainer

    def _run_online_training_iteration(self, env):
        if self.iteration_num % self._config.num_steps_per_iteration != 0:
            return

        buffer = ReplayBuffer(capacity=self._config.num_steps_per_iteration)

        num_steps = 0
        while num_steps <= self._config.num_steps_per_iteration:
            experience = self._environment_explorer.rollout(env)
            experience = self._label_experience(experience)
            buffer.append(experience)
            num_steps += len(experience)

        self._gail_training(buffer)

    def _label_experience(self, experience):
        labeled_experience = []
        if not hasattr(self, '_s_var_label'):
            # build graph
            self._s_var_label = nn.Variable((1, *self._env_info.state_shape))
            self._s_next_var_label = nn.Variable((1, *self._env_info.state_shape))
            if self._env_info.is_discrete_action_env():
                self._a_var_label = nn.Variable((1, 1))
            else:
                self._a_var_label = nn.Variable((1, self._env_info.action_dim))
            logits_fake = self._discriminator.r(self._s_var_label, self._a_var_label, self._s_next_var_label)
            self._reward = -NF.log(1. - NF.sigmoid(logits_fake) + 1e-8)

        for s, a, _, non_terminal, n_s, info in experience:
            # forward and get reward
            self._s_var_label.d = s.reshape((1, -1))
            self._a_var_label.d = a.reshape((1, -1))
            self._s_next_var_label.d = n_s.reshape((1, -1))
            self._reward.forward()
            transition = (s, a, self._reward.d, non_terminal, n_s, info)
            labeled_experience.append(transition)

        return labeled_experience

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _gail_training(self, buffer):
        buffer_iterator = BufferIterator(buffer, 1, shuffle=False, repeat=False)

        # policy learning
        if self._iteration_num % self._config.policy_update_frequency == 0:
            s, a, v_target, advantage = self._align_policy_experiences(buffer_iterator)

            if self._config.preprocess_state:
                self._pi_v_state_preprocessor.update(s)

            self._policy_training(s, a, v_target, advantage)
            self._v_function_training(s, v_target)

        # discriminator learning
        if self._iteration_num % self._config.discriminator_update_frequency == 0:
            s_curr_expert, a_curr_expert, s_next_expert, s_curr_agent, a_curr_agent, s_next_agent = \
                self._align_discriminator_experiences(buffer_iterator)

            if self._config.preprocess_state:
                self._r_state_preprocessor.update(np.concatenate([s_curr_agent, s_curr_expert], axis=0))

            self._discriminator_training(s_curr_expert, a_curr_expert, s_next_expert,
                                         s_curr_agent, a_curr_agent, s_next_agent)

    def _align_policy_experiences(self, buffer_iterator):
        v_target_batch, adv_batch = self._compute_v_target_and_advantage(buffer_iterator)

        s_batch, a_batch, _ = self._align_state_and_action(buffer_iterator)

        return s_batch[:self._config.num_steps_per_iteration], \
            a_batch[:self._config.num_steps_per_iteration], \
            v_target_batch[:self._config.num_steps_per_iteration], \
            adv_batch[:self._config.num_steps_per_iteration]

    def _compute_v_target_and_advantage(self, buffer_iterator):
        v_target_batch = []
        adv_batch = []

        buffer_iterator.reset()
        for experiences, *_ in buffer_iterator:
            # length of experiences is 1
            v_target, adv = compute_v_target_and_advantage(
                self._v_function, experiences[0], gamma=self._config.gamma, lmb=self._config.lmb)
            v_target_batch.append(v_target.reshape(-1, 1))
            adv_batch.append(adv.reshape(-1, 1))

        adv_batch = np.concatenate(adv_batch, axis=0)
        v_target_batch = np.concatenate(v_target_batch, axis=0)

        adv_mean = np.mean(adv_batch)
        adv_std = np.std(adv_batch)
        adv_batch = (adv_batch - adv_mean) / adv_std
        return v_target_batch, adv_batch

    def _align_state_and_action(self, buffer_iterator, batch_size=None):
        s_batch = []
        a_batch = []
        s_next_batch = []

        buffer_iterator.reset()
        for experiences, _ in buffer_iterator:
            # length of experiences is 1
            s_seq, a_seq, _, _, s_next_seq, *_ = marshal_experiences(experiences[0])
            s_batch.append(s_seq)
            a_batch.append(a_seq)
            s_next_batch.append(s_next_seq)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        s_next_batch = np.concatenate(s_next_batch, axis=0)

        if batch_size is None:
            return s_batch, a_batch, s_next_batch

        idx = random.sample(list(range(s_batch.shape[0])), batch_size)
        return s_batch[idx], a_batch[idx], s_next_batch[idx]

    def _align_discriminator_experiences(self, buffer_iterator):
        # sample expert data
        expert_experience, _ = self._expert_buffer.sample(self._config.discriminator_batch_size)
        s_expert_batch, a_expert_batch, _, _, s_next_expert_batch, *_ = marshal_experiences(expert_experience)
        # sample agent data
        s_batch, a_batch, s_next_batch = self._align_state_and_action(
            buffer_iterator, batch_size=self._config.discriminator_batch_size)

        return s_expert_batch, a_expert_batch, s_next_expert_batch, s_batch, a_batch, s_next_batch

    def _v_function_training(self, s, v_target):
        num_iterations_per_epoch = self._config.num_steps_per_iteration // self._config.vf_batch_size

        for _ in range(self._config.vf_epochs * num_iterations_per_epoch):
            indices = np.random.randint(0, self._config.num_steps_per_iteration, size=self._config.vf_batch_size)
            batch = TrainingBatch(batch_size=self._config.vf_batch_size,
                                  s_current=s[indices],
                                  extra={'v_target': v_target[indices]})
            self._v_function_trainer_state = self._v_function_trainer.train(batch)

    def _policy_training(self, s, a, v_target, advantage):
        extra = {}
        extra['v_target'] = v_target[:self._config.pi_batch_size]
        extra['advantage'] = advantage[:self._config.pi_batch_size]
        batch = TrainingBatch(batch_size=self._config.pi_batch_size,
                              s_current=s[:self._config.pi_batch_size],
                              a_current=a[:self._config.pi_batch_size],
                              extra=extra)

        self._policy_trainer_state = self._policy_trainer.train(batch)

    def _discriminator_training(self, s_curr_expert, a_curr_expert, s_next_expert,
                                s_curr_agent, a_curr_agent, s_next_agent):
        extra = {}
        extra['s_current_agent'] = s_curr_agent[:self._config.discriminator_batch_size]
        extra['a_current_agent'] = a_curr_agent[:self._config.discriminator_batch_size]
        extra['s_next_agent'] = s_next_agent[:self._config.discriminator_batch_size]
        extra['s_current_expert'] = s_curr_expert[:self._config.discriminator_batch_size]
        extra['a_current_expert'] = a_curr_expert[:self._config.discriminator_batch_size]
        extra['s_next_expert'] = s_next_expert[:self._config.discriminator_batch_size]

        batch = TrainingBatch(batch_size=self._config.discriminator_batch_size,
                              extra=extra)

        self._discriminator_trainer_state = self._discriminator_trainer.train(batch)

    @eval_api
    def _compute_action(self, s, act_deterministic=False):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._eval_a_distribution = self._policy.pi(self._eval_state_var)

        if act_deterministic:
            eval_a = self._deterministic_action()
        else:
            eval_a = self._probabilistic_action()

        self._eval_state_var.d = s
        eval_a.forward()
        return np.squeeze(eval_a.d, axis=0), {}

    def _deterministic_action(self):
        if not hasattr(self, '_eval_deterministic_a'):
            self._eval_deterministic_a = self._eval_a_distribution.choose_probable()
        return self._eval_deterministic_a

    def _probabilistic_action(self):
        if not hasattr(self, '_eval_probabilistic_a'):
            self._eval_probabilistic_a = self._eval_a_distribution.sample()
        return self._eval_probabilistic_a

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._v_function.scope_name] = self._v_function
        models[self._discriminator.scope_name] = self._discriminator
        if self._config.preprocess_state and isinstance(self._r_state_preprocessor, Model):
            models[self._r_state_preprocessor.scope_name] = self._r_state_preprocessor
        if self._config.preprocess_state and isinstance(self._pi_v_state_preprocessor, Model):
            models[self._pi_v_state_preprocessor.scope_name] = self._pi_v_state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._v_function.scope_name] = self._v_function_solver
        solvers[self._discriminator.scope_name] = self._discriminator_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(GAIL, self).latest_iteration_state
        if hasattr(self, '_discriminator_trainer_state'):
            latest_iteration_state['scalar'].update({'reward_loss': self._discriminator_trainer_state['reward_loss']})
        if hasattr(self, '_v_function_trainer_state'):
            latest_iteration_state['scalar'].update({'v_loss': self._v_function_trainer_state['v_loss']})
        return latest_iteration_state
