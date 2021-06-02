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
from typing import Any, Dict, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedEnvironmentException
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import QRDQNQuantileDistributionFunction, QuantileDistributionFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class QRDQNConfig(AlgorithmConfig):
    '''
    List of configurations for QRDQN algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00005.
        batch_size (int): training atch size. Defaults to 32.
        learner_update_frequency (int): the interval of learner update. Defaults to 4.
        target_update_frequency (int): the interval of target q-function update. Defaults to 10000.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.
            Defaults to 50000.
        replay_buffer_size (int): the capacity of replay buffer. Defaults to 1000000.
        max_explore_steps (int): the number of steps decaying the epsilon value.\
            The epsilon will be decayed linearly \
            :math:`\\epsilon=\\epsilon_{init} - step\\times\\frac{\\epsilon_{init} - \
            \\epsilon_{final}}{max\\_explore\\_steps}`.\
            Defaults to 1000000.
        initial_epsilon (float): the initial epsilon value for ε-greedy explorer. Defaults to 1.0.
        final_epsilon (float): the last epsilon value for ε-greedy explorer. Defaults to 0.01.
        test_epsilon (float): the epsilon value on testing. Defaults to 0.001.
        num_quantiles (int): Number of quantile points. Defaults to 200.
        kappa (float): threshold value of quantile huber loss. Defaults to 1.0.
    '''

    gamma: float = 0.99
    learning_rate: float = 0.00005
    batch_size: int = 32
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    max_explore_steps: int = 1000000
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    num_quantiles: int = 200
    kappa: float = 1.0

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
        self._assert_positive(self.num_quantiles, 'num_quantiles')
        self._assert_positive(self.kappa, 'kappa')


class DefaultQuantileBuilder(ModelBuilder[QuantileDistributionFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: QRDQNConfig,
                    **kwargs) -> QuantileDistributionFunction:
        return QRDQNQuantileDistributionFunction(scope_name, env_info.action_dim, algorithm_config.num_quantiles)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: QRDQNConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate, eps=1e-2 / algorithm_config.batch_size)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: QRDQNConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class QRDQN(Algorithm):
    '''Quantile Regression DQN algorithm.

    This class implements the Quantile Regression DQN algorithm
    proposed by W. Dabney, et al. in the paper: "Distributional Reinforcement Learning with Quantile Regression"
    For details see: https://arxiv.org/abs/1710.10044

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`QRDQNConfig <nnabla_rl.algorithms.qrdqn.QRDQNConfig>`):
            configuration of QRDQN algorithm
        quantile_dist_function_builder (:py:class:`ModelBuilder[QuantileDistributionFunction] \
            <nnabla_rl.builders.ModelBuilder>`): builder of quantile distribution function models
        quantile_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for quantile distribution function solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: QRDQNConfig

    _quantile_dist: QuantileDistributionFunction
    _quantile_dist_solver: nn.solver.Solver
    _target_quantile_dist: QuantileDistributionFunction

    _replay_buffer: ReplayBuffer
    _environment_explorer: EnvironmentExplorer
    _quantile_dist_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _a_greedy: nn.Variable

    _quantile_dist_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: QRDQNConfig = QRDQNConfig(),
                 quantile_dist_function_builder: ModelBuilder[QuantileDistributionFunction] = DefaultQuantileBuilder(),
                 quantile_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(QRDQN, self).__init__(env_or_env_info, config=config)
        if not self._env_info.is_discrete_action_env():
            raise UnsupportedEnvironmentException('{} only supports discrete action environment'.format(self.__name__))

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._quantile_dist = quantile_dist_function_builder('quantile_dist_train', self._env_info, self._config)
            self._quantile_dist_solver = quantile_solver_builder(self._env_info, self._config)
            self._target_quantile_dist = cast(QuantileDistributionFunction,
                                              self._quantile_dist.deepcopy('quantile_dist_target'))

            self._replay_buffer = replay_buffer_builder(self._env_info, self._config)

    @eval_api
    def compute_eval_action(self, state):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            (action, _), _ = epsilon_greedy_action_selection(state,
                                                             self._greedy_action_selector,
                                                             self._random_action_selector,
                                                             epsilon=self._config.test_epsilon)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._quantile_dist_trainer = self._setup_quantile_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            warmup_random_steps=self._config.start_timesteps,
            initial_step_num=self.iteration_num,
            initial_epsilon=self._config.initial_epsilon,
            final_epsilon=self._config.final_epsilon,
            max_explore_steps=self._config.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=self._greedy_action_selector,
            random_action_selector=self._random_action_selector,
            env_info=self._env_info,
            config=explorer_config)
        return explorer

    def _setup_quantile_function_training(self, env_or_buffer):
        trainer_config = MT.q_value.QRDQNQTrainerConfig(
            num_quantiles=self._config.num_quantiles,
            kappa=self._config.kappa)

        quantile_dist_trainer = MT.q_value_trainers.QRDQNQTrainer(
            train_functions=self._quantile_dist,
            solvers={self._quantile_dist.scope_name: self._quantile_dist_solver},
            target_function=self._target_quantile_dist,
            env_info=self._env_info,
            config=trainer_config)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        sync_model(self._quantile_dist, self._target_quantile_dist)
        return quantile_dist_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._config.start_timesteps < self.iteration_num:
            if self.iteration_num % self._config.learner_update_frequency == 0:
                self._qrdqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._qrdqn_training(buffer)

    def _qrdqn_training(self, replay_buffer):
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

        self._quantile_dist_trainer_state = self._quantile_dist_trainer.train(batch)
        if self.iteration_num % self._config.target_update_frequency:
            sync_model(self._quantile_dist, self._target_quantile_dist)

    @eval_api
    def _greedy_action_selector(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            q_function = self._quantile_dist.as_q_function()
            self._a_greedy = q_function.argmax_q(self._eval_state_var)
        self._eval_state_var.d = s
        self._a_greedy.forward()
        return np.squeeze(self._a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _models(self):
        models = {}
        models[self._quantile_dist.scope_name] = self._quantile_dist
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._quantile_dist.scope_name] = self._quantile_dist_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(QRDQN, self).latest_iteration_state
        if hasattr(self, '_quantile_dist_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': self._quantile_dist_trainer_state['q_loss']})
        return latest_iteration_state
