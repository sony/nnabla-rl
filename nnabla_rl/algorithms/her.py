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
from typing import Optional, Tuple, Union

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl as rl
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
import nnabla_rl.preprocessors as RP
from nnabla_rl.algorithm import eval_api
from nnabla_rl.algorithms import DDPG, DDPGConfig
from nnabla_rl.algorithms.common_utils import _StatePreprocessedPolicy, _StatePreprocessedQFunction
from nnabla_rl.builders import ModelBuilder, PreprocessorBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import DeterministicPolicy, HERPolicy, HERQFunction, Model, QFunction
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.replay_buffers.hindsight_replay_buffer import HindsightReplayBuffer
from nnabla_rl.utils.data import add_batch_dimension, marshal_experiences, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, sync_model


@dataclass
class HERConfig(DDPGConfig):
    '''HERConfig
    List of configurations for HER algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size(int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        exploration_noise_sigma (float): standard deviation of gaussian exploration noise. Defaults to 0.1.
        n_cycles (int): the number of cycle. \
            A cycle means collecting experiences for some episodes and updating model for several times.
        n_rollout (int): the number of episode in which policy collect experiences.
        n_update (int): the number of updating model
        max_timesteps (int): the timestep when finishing one epsode.
        hindsight_prob (float): the probability at which buffer samples hindsight goal.
        action_loss_coef (float): the value of coefficient about action loss in policy trainer.
        return_clip (Optional[Tuple[float, float]]): the range of clipping return value.
        exploration_epsilon (float): the value for ε-greedy explorer.
        preprocess_state (bool): Enable preprocessing the states in the collected experiences\
            before feeding as training batch. Defaults to True.
        normalize_epsilon (float): the minimum value of standard deviation of preprocessed state.
        normalize_clip_range (Optional[Tuple[float, float]]): the range of clipping state.
        observation_clip_range (Optional[Tuple[float, float]]): the range of clipping observation.
    '''

    n_cycles: int = 50
    n_rollout: int = 16
    n_update: int = 40
    max_timesteps: int = 50
    hindsight_prob: float = 0.8
    action_loss_coef: float = 1.0
    return_clip: Optional[Tuple[float, float]] = (-50.0, 0.0)
    exploration_epsilon: float = 0.3
    preprocess_state: bool = True
    normalize_epsilon: float = 0.01
    normalize_clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0)
    observation_clip_range: Optional[Tuple[float, float]] = (-200.0, 200.0)


class HERActorBuilder(ModelBuilder[DeterministicPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: HERConfig,
                    **kwargs) -> DeterministicPolicy:
        max_action_value = float(env_info.action_space.high[0])
        return HERPolicy(scope_name, env_info.action_dim, max_action_value=max_action_value)


class HERCriticBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: HERConfig,
                    **kwargs) -> QFunction:
        target_policy = kwargs.get('target_policy')
        return HERQFunction(scope_name, optimal_policy=target_policy)


class HERPreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(self,  # type: ignore[override]
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_config: HERConfig,
                           **kwargs) -> Preprocessor:
        return RP.HERPreprocessor('preprocessor', env_info.state_shape,
                                  epsilon=algorithm_config.normalize_epsilon,
                                  value_clip=algorithm_config.normalize_clip_range)


class HERSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: HERConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class HindsightReplayBufferBuilder(ReplayBufferBuilder):
    def __call__(self, env_info, algorithm_config, **kwargs):
        return HindsightReplayBuffer(reward_function=env_info.reward_function,
                                     hindsight_prob=algorithm_config.hindsight_prob,
                                     capacity=algorithm_config.replay_buffer_size)


class HER(DDPG):
    '''Hindsight Experience Replay (HER) algorithm implementation.

    This class implements the Hindsight Experience Replay (HER) algorithm
    proposed by M. Andrychowicz, et al. in the paper: "Hindsight Experience Replay"
    For detail see: https://arxiv.org/abs/1707.06347

    This algorithm only supports online training.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`HERConfig <nnabla_rl.algorithms.her.HERConfig>`): configuration of HER algorithm
        critic_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of critic models
        critic_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for critic solvers
        actor_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        actor_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for actor solvers
        state_preprocessor_builder (None or :py:class:`PreprocessorBuilder <nnabla_rl.builders.PreprocessorBuilder>`):
            state preprocessor builder to preprocess the states
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
    '''
    _config: HERConfig
    _q: QFunction
    _q_solver: nn.solver.Solver
    _target_q: QFunction
    _pi: DeterministicPolicy
    _pi_solver: nn.solver.Solver
    _target_pi: DeterministicPolicy
    _state_preprocessor: Optional[Preprocessor]
    _replay_buffer: HindsightReplayBuffer

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: HERConfig = HERConfig(),
                 critic_builder: ModelBuilder[QFunction] = HERCriticBuilder(),
                 critic_solver_builder: SolverBuilder = HERSolverBuilder(),
                 actor_builder: ModelBuilder[DeterministicPolicy] = HERActorBuilder(),
                 actor_solver_builder: SolverBuilder = HERSolverBuilder(),
                 state_preprocessor_builder: Optional[PreprocessorBuilder] = HERPreprocessorBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = HindsightReplayBufferBuilder()):

        super(HER, self).__init__(env_or_env_info=env_or_env_info,
                                  config=config,
                                  critic_builder=critic_builder,
                                  critic_solver_builder=critic_solver_builder,
                                  actor_builder=actor_builder,
                                  actor_solver_builder=actor_solver_builder,
                                  replay_buffer_builder=replay_buffer_builder)

        if self._config.preprocess_state and state_preprocessor_builder is not None:
            preprocessor = state_preprocessor_builder('preprocessor', self._env_info, self._config)
            assert preprocessor is not None
            self._q = _StatePreprocessedQFunction(q_function=self._q, preprocessor=preprocessor)
            self._target_q = _StatePreprocessedQFunction(q_function=self._target_q, preprocessor=preprocessor)
            self._pi = \
                _StatePreprocessedPolicy(policy=self._pi, preprocessor=preprocessor)  # type: ignore
            self._target_pi = \
                _StatePreprocessedPolicy(policy=self._target_pi, preprocessor=preprocessor)  # type: ignore
            self._state_preprocessor = preprocessor

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_config = MT.q_value.HERQTrainerConfig(reduction_method='mean',
                                                                 grad_clip=None,
                                                                 return_clip=self._config.return_clip)

        q_function_trainer = MT.q_value.HERQTrainer(train_functions=self._q,
                                                    solvers={self._q.scope_name: self._q_solver},
                                                    target_functions=self._target_q,
                                                    target_policy=self._target_pi,
                                                    env_info=self._env_info,
                                                    config=q_function_trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = \
            MT.policy_trainers.HERPolicyTrainerConfig(action_loss_coef=self._config.action_loss_coef)
        policy_trainer = MT.policy_trainers.HERPolicyTrainer(models=self._pi,
                                                             solvers={self._pi.scope_name: self._pi_solver},
                                                             q_function=self._q,
                                                             env_info=self._env_info,
                                                             config=policy_trainer_config)
        sync_model(self._pi, self._target_pi)
        return policy_trainer

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None

        epsilon_greedy_explorer_config = EE.NoDecayEpsilonGreedyExplorerConfig(
            epsilon=self._config.exploration_epsilon,
            warmup_random_steps=self._config.start_timesteps,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False,
        )
        epsilon_greedy_explorer = EE.NoDecayEpsilonGreedyExplorer(
            greedy_action_selector=self._compute_greedy_with_gaussian_action,
            random_action_selector=self._compute_random_action,
            env_info=self._env_info,
            config=epsilon_greedy_explorer_config,
        )
        return epsilon_greedy_explorer

    def _run_online_training_iteration(self, env):
        iteration_per_epoch = self._config.max_timesteps * self._config.n_cycles * self._config.n_update
        if self.iteration_num % iteration_per_epoch != 0:
            return

        for _ in range(self._config.n_cycles):
            self._collect_experiences(env)

            if self._config.batch_size < len(self._replay_buffer):
                self._her_training(self._replay_buffer)

    def _collect_experiences(self, env):
        for _ in range(self._config.n_rollout):
            experiences = self._environment_explorer.rollout(env)
            experiences = experiences[:-1]
            if self._config.preprocess_state:
                state, *_ = marshal_experiences(experiences)
                state = self._hindsight_state(state)
                self._state_preprocessor.update(state)
            self._replay_buffer.append_all(experiences)

    def _hindsight_state(self, state):
        observation, goal, achieved_goal = state

        data_num = goal.shape[0]
        goal = self._select_goal(goal, achieved_goal, data_num)
        return (observation, goal, achieved_goal)

    def _select_goal(self, goal, achieved_goal, data_num):
        her_indices = np.where(rl.random.drng.random(data_num) <= self._config.hindsight_prob)[0]
        future_indices = rl.random.drng.integers(her_indices, data_num)
        goal_for_compute_mean_std = goal.copy()
        goal_for_compute_mean_std[her_indices] = achieved_goal[future_indices]
        return goal_for_compute_mean_std

    def _her_training(self, replay_buffer):
        for i in range(self._config.n_update):
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
            self._policy_trainer_state = self._policy_trainer.train(batch)

            td_errors = np.abs(self._q_function_trainer_state['td_errors'])
            replay_buffer.update_priorities(td_errors)

        # target update
        sync_model(self._q, self._target_q, tau=self._config.tau)
        sync_model(self._pi, self._target_pi, tau=self._config.tau)

    @eval_api
    def _compute_greedy_action(self, s):
        # evaluation input/action variables
        s = add_batch_dimension(s)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = create_variable(1, self._env_info.state_shape)
            self._eval_action = self._pi.pi(self._eval_state_var)
        set_data_to_variable(self._eval_state_var, s)
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    @eval_api
    def _compute_greedy_with_gaussian_action(self, s):
        action, info = self._compute_greedy_action(s)
        action_clip_low = self._env_info.action_space.low
        action_clip_high = self._env_info.action_space.high
        action_with_noise = self._append_noise(action, action_clip_low, action_clip_high)
        return action_with_noise, info

    def _append_noise(self, action, low, high):
        sigma = self._config.exploration_noise_sigma
        noise = rl.random.drng.normal(loc=0.0, scale=sigma, size=action.shape).astype(np.float32)
        action_with_noise = np.clip(action + noise, low, high)
        return action_with_noise

    def _compute_random_action(self, s):
        action = self._env_info.action_space.sample()
        return action, {}

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        if self._config.preprocess_state and isinstance(self._state_preprocessor, Model):
            models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info

        # continuous action env
        is_continuous_action_env = env_info.is_continuous_action_env()
        is_goal_conditioned_env = env_info.is_goal_conditioned_env()
        return (is_continuous_action_env and is_goal_conditioned_env)
