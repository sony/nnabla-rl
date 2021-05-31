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
from typing import Any, Dict, List, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedEnvironmentException
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import DeterministicPolicy, QFunction, TD3Policy, TD3QFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class TD3Config(AlgorithmConfig):
    '''TD3Config
    List of configurations for TD3 algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.003.
        batch_size(int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        d (int): Interval of the policy update. The policy will be updated every d q-function updates. Defaults to 2.
        exploration_noise_sigma (float): Standard deviation of the gaussian exploration noise. Defaults to 0.1.
        train_action_noise_sigma (float): Standard deviation of the gaussian action noise used in the training.\
            Defaults to 0.2.
        train_action_noise_abs (float): Absolute limit value of action noise used in the training. Defaults to 0.5.
    '''

    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    tau: float = 0.005
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    d: int = 2
    exploration_noise_sigma: float = 0.1
    train_action_noise_sigma: float = 0.2
    train_action_noise_abs: float = 0.5

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.d, 'd')
        self._assert_positive(self.exploration_noise_sigma, 'exploration_noise_sigma')
        self._assert_positive(self.train_action_noise_sigma, 'train_action_noise_sigma')
        self._assert_positive(self.train_action_noise_abs, 'train_action_noise_abs')


class DefaultCriticBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: TD3Config,
                    **kwargs) -> QFunction:
        target_policy = kwargs.get('target_policy')
        return TD3QFunction(scope_name, optimal_policy=target_policy)


class DefaultActorBuilder(ModelBuilder[DeterministicPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: TD3Config,
                    **kwargs) -> DeterministicPolicy:
        max_action_value = float(env_info.action_space.high[0])
        return TD3Policy(scope_name, env_info.action_dim, max_action_value=max_action_value)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: TD3Config,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: TD3Config,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class TD3(Algorithm):
    '''Twin Delayed Deep Deterministic policy gradient (TD3) algorithm.

    This class implements the Twin Delayed Deep Deteministic policy gradien (TD3) algorithm
    proposed by S. Fujimoto, et al. in the paper: "Addressing Function Approximation Error in Actor-Critic Methods"
    For detail see: https://arxiv.org/abs/1802.09477

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`TD3Config <nnabla_rl.algorithms.td3.TD3Config>`):
            configuration of the TD3 algorithm
        critic_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of critic models
        critic_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of critic solvers
        actor_builder (:py:class:`ModelBuilder[DeterministicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        actor_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of actor solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: TD3Config
    _q1: QFunction
    _q2: QFunction
    _train_q_functions: List[QFunction]
    _train_q_solvers: Dict[str, QFunction]
    _target_q_functions: List[QFunction]
    _pi: DeterministicPolicy
    _pi_solver: nn.solvers.Solver
    _target_pi: DeterministicPolicy
    _replay_buffer: ReplayBuffer

    _environment_explorer: EnvironmentExplorer
    _policy_trainer: ModelTrainer
    _q_function_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _eval_action: nn.Variable

    _policy_trainer_state: Dict[str, Any]
    _q_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: TD3Config = TD3Config(),
                 critic_builder: ModelBuilder[QFunction] = DefaultCriticBuilder(),
                 critic_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 actor_builder: ModelBuilder[DeterministicPolicy] = DefaultActorBuilder(),
                 actor_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(TD3, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_discrete_action_env():
            raise UnsupportedEnvironmentException

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._q1 = critic_builder(scope_name="q1", env_info=self._env_info, algorithm_config=self._config)
            self._q2 = critic_builder(scope_name="q2", env_info=self._env_info, algorithm_config=self._config)
            self._train_q_functions = [self._q1, self._q2]
            self._train_q_solvers = {q.scope_name: critic_solver_builder(
                env_info=self._env_info, algorithm_config=self._config) for q in self._train_q_functions}
            self._target_q_functions = [cast(QFunction, q.deepcopy('target_' + q.scope_name))
                                        for q in self._train_q_functions]

            self._pi = actor_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
            self._pi_solver = actor_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._target_pi = cast(DeterministicPolicy, self._pi.deepcopy('target_' + self._pi.scope_name))

            self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)

    @eval_api
    def compute_eval_action(self, state):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._compute_greedy_action(state)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_config = EE.GaussianExplorerConfig(
            warmup_random_steps=self._config.start_timesteps,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False,
            action_clip_low=self._env_info.action_space.low,
            action_clip_high=self._env_info.action_space.high,
            sigma=self._config.exploration_noise_sigma
        )
        explorer = EE.GaussianExplorer(policy_action_selector=self._compute_greedy_action,
                                       env_info=self._env_info,
                                       config=explorer_config)
        return explorer

    def _setup_q_function_training(self, env_or_buffer):
        # training input/loss variables
        q_function_trainer_config = MT.q_value_trainers.TD3QTrainerConfig(
            reduction_method='mean',
            grad_clip=None,
            train_action_noise_sigma=self._config.train_action_noise_sigma,
            train_action_noise_abs=self._config.train_action_noise_abs)
        q_function_trainer = MT.q_value_trainers.TD3QTrainer(
            train_functions=self._train_q_functions,
            solvers=self._train_q_solvers,
            target_functions=self._target_q_functions,
            target_policy=self._target_pi,
            env_info=self._env_info,
            config=q_function_trainer_config)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q)
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.DPGPolicyTrainerConfig()
        policy_trainer = MT.policy_trainers.DPGPolicyTrainer(
            models=self._pi,
            solvers={self._pi.scope_name: self._pi_solver},
            q_function=self._q1,
            env_info=self._env_info,
            config=policy_trainer_config)
        sync_model(self._pi, self._target_pi, 1.0)

        return policy_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)

        if self._config.start_timesteps < self.iteration_num:
            self._td3_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._td3_training(buffer)

    def _td3_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._config.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshal_experiences(experiences)
        batch = TrainingBatch(batch_size=self._config.batch_size,
                              s_current=s,
                              a_current=a,
                              gamma=self._config.gamma,
                              reward=r,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=info['weights'])

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        td_errors = np.abs(self._q_function_trainer_state['td_errors'])
        replay_buffer.update_priorities(td_errors)

        if self.iteration_num % self._config.d == 0:
            # Optimize actor
            self._policy_trainer_state = self._policy_trainer.train(batch)

            # parameter update
            for q, target_q in zip(self._train_q_functions, self._target_q_functions):
                sync_model(q, target_q, tau=self._config.tau)
            sync_model(self._pi, self._target_pi, tau=self._config.tau)

    @eval_api
    def _compute_greedy_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._eval_action = self._pi.pi(self._eval_state_var)
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    def _models(self):
        models = {}
        models[self._q1.scope_name] = self._q1
        models[self._q2.scope_name] = self._q2
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._pi.scope_name] = self._pi_solver
        solvers.update(self._train_q_solvers)
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(TD3, self).latest_iteration_state
        if hasattr(self, '_policy_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_loss': self._policy_trainer_state['pi_loss']})
        if hasattr(self, '_q_function_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': self._q_function_trainer_state['q_loss']})
            latest_iteration_state['histogram'].update(
                {'td_errors': self._q_function_trainer_state['td_errors'].flatten()})
        return latest_iteration_state
