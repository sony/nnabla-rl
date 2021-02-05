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

import gym
import numpy as np

from typing import cast, Callable, Union

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.builders import StateActionQuantileFunctionBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import IQNQuantileFunction, StateActionQuantileFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


@dataclass
class MunchausenIQNParam(AlgorithmParam):
    batch_size: int = 32
    gamma: float = 0.99
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    max_explore_steps: int = 1000000
    learning_rate: float = 0.00005
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    N: int = 64
    N_prime: int = 64
    K: int = 32
    kappa: float = 1.0
    embedding_dim: int = 64

    # munchausen iqn training parameters
    entropy_temperature: float = 0.03
    munchausen_scaling_term: float = 0.9
    clipping_value: float = -1

    def __post_init__(self):
        '''__post_init__

        Check that set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.learner_update_frequency, 'learner_update_frequency')
        self._assert_positive(self.target_update_frequency, 'target_update_frequency')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.initial_epsilon, 'initial_epsilon')
        self._assert_positive(self.final_epsilon, 'final_epsilon')
        self._assert_positive(self.test_epsilon, 'test_epsilon')
        self._assert_positive(self.N, 'N')
        self._assert_positive(self.N_prime, 'N_prime')
        self._assert_positive(self.K, 'K')
        self._assert_positive(self.kappa, 'kappa')
        self._assert_positive(self.embedding_dim, 'embedding_dim')
        self._assert_negative(self.clipping_value, 'clipping_value')


def risk_neutral_measure(tau):
    return tau


class DefaultQuantileFunctionBuilder(StateActionQuantileFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: MunchausenIQNParam,
                    **kwargs) -> StateActionQuantileFunction:
        assert isinstance(algorithm_params, MunchausenIQNParam)
        risk_measure_function = kwargs['risk_measure_function']
        return IQNQuantileFunction(scope_name,
                                   env_info.action_dim,
                                   algorithm_params.embedding_dim,
                                   K=algorithm_params.K,
                                   risk_measure_function=risk_measure_function)


class DefaultQuantileSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_params: MunchausenIQNParam,
                     **kwargs) -> nn.solvers.Solver:
        assert isinstance(algorithm_params, MunchausenIQNParam)
        return NS.Adam(algorithm_params.learning_rate, eps=1e-2 / algorithm_params.batch_size)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_params: MunchausenIQNParam,
                            **kwargs) -> ReplayBuffer:
        assert isinstance(algorithm_params, MunchausenIQNParam)
        return ReplayBuffer(capacity=algorithm_params.replay_buffer_size)


class MunchausenIQN(Algorithm):
    '''Munchausen-IQN algorithm implementation.

    This class implements the Munchausen-IQN (Munchausen Implicit Quantile Network) algorithm
    proposed by N. Vieillard, et al. in the paper: "Munchausen Reinforcement Learning"
    For detail see: https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf
    '''

    _params: MunchausenIQNParam
    _quantile_function: StateActionQuantileFunction
    _target_quantile_function: StateActionQuantileFunction
    _quantile_function_solver: nn.solver.Solver
    _replay_buffer: ReplayBuffer

    _environment_explorer: EnvironmentExplorer
    _quantile_function_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _a_greedy: nn.Variable

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 params: MunchausenIQNParam = MunchausenIQNParam(),
                 risk_measure_function: Callable[[nn.Variable], nn.Variable] = risk_neutral_measure,
                 quantile_function_builder: StateActionQuantileFunctionBuilder = DefaultQuantileFunctionBuilder(),
                 quantile_solver_builder: SolverBuilder = DefaultQuantileSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(MunchausenIQN, self).__init__(env_or_env_info, params=params)

        if not self._env_info.is_discrete_action_env():
            raise ValueError('{} only supports discrete action environment'.format(self.__name__))

        kwargs = {}
        kwargs['risk_measure_function'] = risk_measure_function
        self._quantile_function = quantile_function_builder('quantile_function', self._env_info, self._params, **kwargs)
        self._target_quantile_function = cast(StateActionQuantileFunction,
                                              self._quantile_function.deepcopy('target_quantile_function'))

        self._quantile_function_solver = quantile_solver_builder(self._env_info, self._params)

        self._replay_buffer = replay_buffer_builder(self._env_info, self._params)

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._quantile_function_trainer = self._setup_quantile_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_params = EE.LinearDecayEpsilonGreedyExplorerParam(
            warmup_random_steps=self._params.start_timesteps,
            initial_step_num=self.iteration_num,
            initial_epsilon=self._params.initial_epsilon,
            final_epsilon=self._params.final_epsilon,
            max_explore_steps=self._params.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=self._greedy_action_selector,
            random_action_selector=self._random_action_selector,
            env_info=self._env_info,
            params=explorer_params)
        return explorer

    def _setup_quantile_function_training(self, env_or_buffer):
        trainer_params = MT.q_value_trainers.IQNQuantileFunctionTrainerParam(
            N=self._params.N,
            N_prime=self._params.N_prime,
            K=self._params.K,
            kappa=self._params.kappa)

        quantile_function_trainer = MT.q_value_trainers.IQNQuantileFunctionTrainer(
            self._env_info,
            params=trainer_params)

        target_update_frequency = self._params.target_update_frequency // self._params.learner_update_frequency
        training = MT.q_value_trainings.MunchausenRLTraining(train_function=self._quantile_function,
                                                             target_function=self._target_quantile_function,
                                                             tau=self._params.entropy_temperature,
                                                             alpha=self._params.munchausen_scaling_term,
                                                             clip_min=self._params.clipping_value,
                                                             clip_max=0.0)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._quantile_function,
            dst_models=self._target_quantile_function,
            target_update_frequency=target_update_frequency,
            tau=1.0)
        quantile_function_trainer.setup_training(
            self._quantile_function, {self._quantile_function.scope_name: self._quantile_function_solver}, training)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        copy_network_parameters(self._quantile_function.get_parameters(),
                                self._target_quantile_function.get_parameters())

        return quantile_function_trainer

    @eval_api
    def compute_eval_action(self, state):
        (action, _), _ = epsilon_greedy_action_selection(state,
                                                         self._greedy_action_selector,
                                                         self._random_action_selector,
                                                         epsilon=self._params.test_epsilon)
        return action

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._params.start_timesteps < self.iteration_num:
            if self.iteration_num % self._params.learner_update_frequency == 0:
                self._m_iqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._m_iqn_training(buffer)

    def _m_iqn_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshall_experiences(experiences)
        batch = TrainingBatch(batch_size=self._params.batch_size,
                              s_current=s,
                              a_current=a,
                              gamma=self._params.gamma,
                              reward=r,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=info['weights'])

        self._quantile_function_trainer.train(batch)

    def _greedy_action_selector(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            q_function = self._quantile_function.as_q_function()
            self._a_greedy = q_function.argmax_q(self._eval_state_var)
        self._eval_state_var.d = s
        self._a_greedy.forward()
        return np.squeeze(self._a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _models(self):
        models = {}
        models[self._quantile_function.scope_name] = self._quantile_function
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._quantile_function.scope_name] = self._quantile_function_solver
        return solvers
