# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import nnabla.solvers as NS


from dataclasses import dataclass

import numpy as np

import gym
from typing import cast, Union

from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import QFunctionBuilder, DeterministicPolicyBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, Training
from nnabla_rl.models import TD3QFunction, TD3Policy, QFunction, DeterministicPolicy
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


@dataclass
class DDPGConfig(AlgorithmConfig):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    exploration_noise_sigma: float = 0.1


class DefaultCriticBuilder(QFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DDPGConfig,
                    **kwargs) -> QFunction:
        target_policy = kwargs.get('target_policy')
        return TD3QFunction(scope_name, optimal_policy=target_policy)


class DefaultActorBuilder(DeterministicPolicyBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DDPGConfig,
                    **kwargs) -> DeterministicPolicy:
        max_action_value = float(env_info.action_space.high[0])
        return TD3Policy(scope_name, env_info.action_dim, max_action_value=max_action_value)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DDPGConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: DDPGConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DDPG(Algorithm):
    _config: DDPGConfig
    _q: QFunction
    _q_solver: nn.solver.Solver
    _target_q: QFunction
    _pi: DeterministicPolicy
    _pi_solver: nn.solver.Solver
    _target_pi: DeterministicPolicy
    _replay_buffer: ReplayBuffer
    _environment_explorer: EnvironmentExplorer
    _q_function_trainer: ModelTrainer
    _policy_trainer: ModelTrainer

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DDPGConfig = DDPGConfig(),
                 critic_builder: QFunctionBuilder = DefaultCriticBuilder(),
                 critic_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 actor_builder: DeterministicPolicyBuilder = DefaultActorBuilder(),
                 actor_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(DDPG, self).__init__(env_or_env_info, config=config)

        self._q = critic_builder(scope_name="q", env_info=self._env_info, algorithm_config=self._config)
        self._q_solver = critic_solver_builder(env_info=self._env_info, algorithm_config=self._config)
        self._target_q = cast(QFunction, self._q.deepcopy('target_' + self._q.scope_name))

        self._pi = actor_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
        self._pi_solver = actor_solver_builder(env_info=self._env_info, algorithm_config=self._config)
        self._target_pi = cast(DeterministicPolicy, self._pi.deepcopy("target_" + self._pi.scope_name))

        self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)

    def _before_training_start(self, env_or_buffer):
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
        q_function_trainer_config = MT.q_value_trainers.SquaredTDQFunctionTrainerConfig(
            reduction_method='mean',
            grad_clip=None)

        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            config=q_function_trainer_config)

        training = MT.q_value_trainings.DDPGTraining(
            train_functions=self._q,
            target_functions=self._target_q,
            target_policy=self._target_pi)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._q,
            dst_models=self._target_q,
            target_update_frequency=1,
            tau=self._config.tau)
        q_function_trainer.setup_training(self._q, {self._q.scope_name: self._q_solver}, training)
        copy_network_parameters(self._q.get_parameters(), self._target_q.get_parameters())
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.DPGPolicyTrainerConfig()
        policy_trainer = MT.policy_trainers.DPGPolicyTrainer(env_info=self._env_info,
                                                             config=policy_trainer_config,
                                                             q_function=self._q)
        # Empty training will not configure anything and does the default training written in policy_trainer class
        training = Training()
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._pi,
            dst_models=self._target_pi,
            target_update_frequency=1,
            tau=self._config.tau
        )
        policy_trainer.setup_training(self._pi, {self._pi.scope_name: self._pi_solver}, training)
        copy_network_parameters(self._pi.get_parameters(), self._target_pi.get_parameters(), tau=1.0)

        return policy_trainer

    @eval_api
    def compute_eval_action(self, state):
        action, _ = self._compute_greedy_action(state)
        return action

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._config.start_timesteps < self.iteration_num:
            self._ddpg_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._ddpg_training(buffer)

    def _ddpg_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._config.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshall_experiences(experiences)
        batch = TrainingBatch(batch_size=self._config.batch_size,
                              s_current=s,
                              a_current=a,
                              gamma=self._config.gamma,
                              reward=r,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=info['weights'])

        errors = self._q_function_trainer.train(batch)
        self._policy_trainer.train(batch)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _compute_greedy_action(self, s):
        # evaluation input/action variables
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._eval_action = self._pi.pi(self._eval_state_var)
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._pi.scope_name] = self._pi_solver
        solvers[self._q.scope_name] = self._q_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['iteration'] = self._iteration_num
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}

        latest_iteration_state['histogram'].update(self._q.get_parameters())

        return latest_iteration_state
