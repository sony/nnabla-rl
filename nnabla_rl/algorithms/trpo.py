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
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
import nnabla_rl.preprocessors as RP
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import (_StatePreprocessedPolicy, _StatePreprocessedVFunction,
                                               compute_v_target_and_advantage)
from nnabla_rl.builders import ModelBuilder, PreprocessorBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import Model, StochasticPolicy, TRPOPolicy, TRPOVFunction, VFunction
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences


@dataclass
class TRPOConfig(AlgorithmConfig):
    '''TRPO config
    Args:
        gamma (float): Discount factor of rewards. Defaults to 0.995.
        lmb (float): Scalar of lambda return's computation in GAE. Defaults to 0.97.\
            This configuration is related to bias and variance of estimated value. \
            If it is close to 0, estimated value is low-variance but biased.\
            If it is close to 1, estimated value is unbiased but high-variance.
        num_steps_per_iteration (int): Number of steps per each training iteration for collecting on-policy experinces.\
            Increasing this step size is effective to get precise parameters of policy and value function updating, \
            but computational time of each iteration will increase. Defaults to 5000.
        pi_batch_size (int): Trainig batch size of policy. \
            Usually, pi_batch_size is the same as num_steps_per_iteration. Defaults to 5000.
        sigma_kl_divergence_constraint (float): Constraint size of kl divergence \
            between previous policy and updated policy. Defaults to 0.01.
        maximum_backtrack_numbers (int): Maximum backtrack numbers of linesearch. Defaults to 10.
        conjugate_gradient_damping (float): Damping size of conjugate gradient method. Defaults to 0.1.
        conjugate_gradient_iterations (int): Number of iterations of conjugate gradient method. Defaults to 20.
        vf_epochs (int): Number of epochs in each iteration. Defaults to 5.
        vf_batch_size (int): Training batch size of value function. Defaults to 64.
        vf_learning_rate (float): Learning rate which is set to the solvers of value function. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        preprocess_state (bool): Enable preprocessing the states in the collected experiences \
            before feeding as training batch. Defaults to True.
        gpu_batch_size (int, optional): Actual batch size to reduce one forward gpu calculation memory. \
            As long as gpu memory size is enough, this configuration should not be specified. If not specified,  \
            gpu_batch_size is the same as pi_batch_size. Defaults to None.
    '''
    gamma: float = 0.995
    lmb: float = 0.97
    num_steps_per_iteration: int = 5000
    pi_batch_size: int = 5000
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 20
    vf_epochs: int = 5
    vf_batch_size: int = 64
    vf_learning_rate: float = 1e-3
    preprocess_state: bool = True
    gpu_batch_size: Optional[int] = None

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.pi_batch_size, 0, self.num_steps_per_iteration, 'pi_batch_size')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.lmb, 0.0, 1.0, 'lmb')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_between(self.pi_batch_size, 0, self.num_steps_per_iteration, 'pi_batch_size')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')
        self._assert_positive(self.vf_epochs, 'vf_epochs')
        self._assert_positive(self.vf_batch_size, 'vf_batch_size')
        self._assert_positive(self.vf_learning_rate, 'vf_learning_rate')


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: TRPOConfig,
                    **kwargs) -> StochasticPolicy:
        return TRPOPolicy(scope_name, env_info.action_dim)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: TRPOConfig,
                    **kwargs) -> VFunction:
        return TRPOVFunction(scope_name)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: TRPOConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.vf_learning_rate)


class DefaultPreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(self,  # type: ignore[override]
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_config: TRPOConfig,
                           **kwargs) -> Preprocessor:
        return RP.RunningMeanNormalizer(scope_name, env_info.state_shape, value_clip=(-5.0, 5.0))


class TRPO(Algorithm):
    '''Trust Region Policy Optimiation method with Generalized Advantage Estimation (GAE) implementation.

    This class implements the Trust Region Policy Optimiation (TRPO)
    with Generalized Advantage Estimation (GAE) algorithm proposed by J. Schulman, et al.
    in the paper: "Trust Region Policy Optimization" and
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    For detail see: https://arxiv.org/abs/1502.05477 and https://arxiv.org/abs/1506.02438

    This algorithm only supports online training.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`PPOConfig <nnabla_rl.algorithms.trpo.TRPOConfig>`): configuration of TRPO algorithm
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for v function solvers
        policy_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
        state_preprocessor_builder (None or :py:class:`PreprocessorBuilder <nnabla_rl.builders.PreprocessorBuilder>`):
            state preprocessor builder to preprocess the states
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: TRPOConfig
    _policy: StochasticPolicy
    _v_function: VFunction
    _v_function_solver: nn.solvers.Solver
    _state_preprocessor: Optional[Preprocessor]
    _environment_explorer: EnvironmentExplorer
    _policy_trainer: ModelTrainer
    _v_function_trainer: ModelTrainer
    _eval_state_var: nn.Variable
    _eval_action: nn.Variable

    _policy_trainer_state: Dict[str, Any]
    _v_function_trainer_state: Dict[str, Any]

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: TRPOConfig = TRPOConfig(),
                 v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 state_preprocessor_builder: Optional[PreprocessorBuilder] = DefaultPreprocessorBuilder()):
        super(TRPO, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_discrete_action_env():
            raise NotImplementedError

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._v_function = v_function_builder('v', self._env_info, self._config)
            self._policy = policy_builder('pi', self._env_info, self._config)

            self._preprocessor: Optional[Preprocessor] = None
            if self._config.preprocess_state and state_preprocessor_builder is not None:
                preprocessor = state_preprocessor_builder('preprocessor', self._env_info, self._config)
                assert preprocessor is not None
                self._v_function = _StatePreprocessedVFunction(v_function=self._v_function, preprocessor=preprocessor)
                self._policy = _StatePreprocessedPolicy(policy=self._policy, preprocessor=preprocessor)
                self._state_preprocessor = preprocessor
            self._v_function_solver = v_solver_builder(self._env_info, self._config)

    @eval_api
    def compute_eval_action(self, s):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._compute_action(s)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

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
        v_function_trainer_config = MT.v_value.MonteCarloVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=1.0
        )
        v_function_trainer = MT.v_value.MonteCarloVTrainer(
            train_functions=self._v_function,
            solvers={self._v_function.scope_name: self._v_function_solver},
            env_info=self._env_info,
            config=v_function_trainer_config)
        return v_function_trainer

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
        s, a, v_target, advantage = self._align_experiences(buffer_iterator)

        if self._config.preprocess_state:
            self._state_preprocessor.update(s)

        # v function training
        self._v_function_training(s, v_target)

        # policy training
        self._policy_training(s, a, v_target, advantage)

    def _align_experiences(self, buffer_iterator):
        v_target_batch, adv_batch = self._compute_v_target_and_advantage(buffer_iterator)

        s_batch, a_batch = self._align_state_and_action(buffer_iterator)

        return s_batch[:self._config.num_steps_per_iteration], \
            a_batch[:self._config.num_steps_per_iteration], \
            v_target_batch[:self._config.num_steps_per_iteration], \
            adv_batch[:self._config.num_steps_per_iteration]

    def _compute_v_target_and_advantage(self, buffer_iterator):
        v_target_batch = []
        adv_batch = []
        buffer_iterator.reset()
        for experiences, _ in buffer_iterator:
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

    def _align_state_and_action(self, buffer_iterator):
        s_batch = []
        a_batch = []

        buffer_iterator.reset()
        for experiences, _ in buffer_iterator:
            # length of experiences is 1
            s_seq, a_seq, *_ = marshal_experiences(experiences[0])
            s_batch.append(s_seq)
            a_batch.append(a_seq)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        return s_batch, a_batch

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

    @eval_api
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
        models[self._v_function.scope_name] = self._v_function
        if self._config.preprocess_state and isinstance(self._state_preprocessor, Model):
            models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._v_function.scope_name] = self._v_function_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(TRPO, self).latest_iteration_state
        if hasattr(self, '_v_function_trainer_state'):
            latest_iteration_state['scalar'].update({'v_loss': self._v_function_trainer_state['v_loss']})
        return latest_iteration_state
