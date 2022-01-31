# Copyright 2022 Sony Group Corporation.
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
from typing import Any, Dict, Optional, Tuple, Union

import gym

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms import DDQN, DDQNConfig
from nnabla_rl.algorithms.common_utils import _GreedyActionSelector
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer
from nnabla_rl.models import ICRA2018QtOptQFunction, QFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.misc import sync_model


@dataclass
class ICRA2018QtOptConfig(DDQNConfig):
    '''
    List of configurations for DQN algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.9.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size (int): training batch size. Defaults to 64.
        num_steps (int): number of steps for N-step Q targets. Defaults to 1.
        q_loss_scalar (float): scale value for the loss function. Defaults to 0.5.
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
        final_epsilon (float): the last epsilon value for ε-greedy explorer. Defaults to 0.1.
        test_epsilon (float): the epsilon value on testing. Defaults to 0.0.
        grad_clip (Optional[Tuple[float, float]]): Clip the gradient of final layer. Defaults to (-1.0, 1.0).
        unroll_steps (int): Number of steps to unroll tranining network.
            The network will be unrolled even though the provided model doesn't have RNN layers.
            Defaults to 1.
        burn_in_steps (int): Number of burn-in steps to initiaze recurrent layer states during training.
            This flag does not take effect if given model is not an RNN model.
            Defaults to 0.
        reset_rnn_on_terminal (bool): Reset recurrent internal states to zero during training if episode ends.
            This flag does not take effect if given model is not an RNN model.
            Defaults to False.
        cem_initial_mean (Optional[Tuple[float, ...]]): the initial mean of cross entropy method's
            gaussian distribution. Defaults to None.
        cem_initial_variance (Optional[Tuple[float, ...]]): the initial variance of cross entropy method's
            gaussian distribution. Defaults to None.
        cem_sample_size (int): number of candidates at the sampling step of cross entropy method.
            Defaults to 64.
        cem_num_elites (int): number of elites for computing the new gaussian distribution of cross entropy method.
            Defaults to 10.
        cem_alpha (float): parameter for soft updating the mean and variance of the gaussian distribution.
            Defaults to 0.
        cem_num_iterations (int): number of optimization iterations of cross entropy method.
            Defaults to 3.
        random_sample_size (int): number of candidates at the sampling step of random shooting method.
            Defaults to 16.
    '''
    gamma: float = 0.9
    learning_rate: float = 0.001
    batch_size: int = 64
    num_steps: int = 1
    q_loss_scalar: float = 0.5
    # network update
    learner_update_frequency: int = 1
    target_update_frequency: int = 50
    test_epsilon: float = 0.0
    # stochastic optimizations
    cem_initial_mean: Optional[Tuple[float, ...]] = None
    cem_initial_variance: Optional[Tuple[float, ...]] = None
    cem_sample_size: int = 64
    cem_num_elites: int = 10
    cem_alpha: float = 0.0
    cem_num_iterations: int = 3
    random_sample_size: int = 16

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.num_steps, 'num_steps')
        self._assert_positive(self.q_loss_scalar, 'q_loss_scalar')
        self._assert_positive(self.learner_update_frequency, 'learner_update_frequency')
        self._assert_positive(self.target_update_frequency, 'target_update_frequency')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_smaller_than(self.start_timesteps, self.replay_buffer_size, 'start_timesteps')
        self._assert_between(self.initial_epsilon, 0.0, 1.0, 'initial_epsilon')
        self._assert_between(self.final_epsilon, 0.0, 1.0, 'final_epsilon')
        self._assert_between(self.test_epsilon, 0.0, 1.0, 'test_epsilon')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_positive(self.unroll_steps, 'unroll_steps')
        self._assert_positive_or_zero(self.burn_in_steps, 'burn_in_steps')
        self._assert_positive(self.cem_sample_size, 'cem_sample_size')
        self._assert_positive(self.cem_num_elites, 'cem_num_elites')
        self._assert_positive_or_zero(self.cem_alpha, "cem_alpha")
        self._assert_positive(self.cem_num_iterations, 'cem_num_iterations')
        self._assert_positive(self.random_sample_size, 'random_sample_size')


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: ICRA2018QtOptConfig,
                    **kwargs) -> QFunction:
        return ICRA2018QtOptQFunction(scope_name,
                                      env_info.action_dim,
                                      action_high=env_info.action_high,
                                      action_low=env_info.action_low,
                                      cem_initial_mean=algorithm_config.cem_initial_mean,
                                      cem_initial_variance=algorithm_config.cem_initial_variance,
                                      cem_sample_size=algorithm_config.cem_sample_size,
                                      cem_num_elites=algorithm_config.cem_num_elites,
                                      cem_num_iterations=algorithm_config.cem_num_iterations,
                                      cem_alpha=algorithm_config.cem_alpha,
                                      random_sample_size=algorithm_config.random_sample_size)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: ICRA2018QtOptConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: ICRA2018QtOptConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: ICRA2018QtOptConfig,
                       algorithm: "ICRA2018QtOpt",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            initial_step_num=algorithm.iteration_num,
            initial_epsilon=algorithm_config.initial_epsilon,
            final_epsilon=algorithm_config.final_epsilon,
            max_explore_steps=algorithm_config.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=algorithm._exploration_action_selector,
            random_action_selector=algorithm._random_action_selector,
            env_info=env_info,
            config=explorer_config)
        return explorer


class ICRA2018QtOpt(DDQN):
    '''DQN algorithm for a continuous action environment.

    This class implements the Deep Q-Network (DQN) algorithm for a continuous action environment.
    proposed by D Quillen, et al. in the paper: 'Deep Reinforcement Learning for Vision-Based Robotic Grasping:
    A Simulated Comparative Evaluation of Off-Policy Methods'
    For details see: https://arxiv.org/pdf/1802.10264.pdf

    This algorithm is a simple version of QtOpt, referring to https://arxiv.org/abs/1806.10293.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DQNConfig <nnabla_rl.algorithms.dqn.DQNConfig>`):
            the parameter for DQN training
        q_func_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of q function model
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of q function solver
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ICRA2018QtOptConfig
    _q: QFunction
    _q_solver: nn.solver.Solver
    _target_q: QFunction
    _replay_buffer: ReplayBuffer
    _explorer_builder: ExplorerBuilder
    _environment_explorer: EnvironmentExplorer
    _q_function_trainer: ModelTrainer

    _q_function_trainer_state: Dict[str, Any]

    _evaluation_actor: _GreedyActionSelector
    _exploration_actor: _GreedyActionSelector

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: ICRA2018QtOptConfig = ICRA2018QtOptConfig(),
                 q_func_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(ICRA2018QtOpt, self).__init__(env_or_env_info=env_or_env_info,
                                            config=config,
                                            q_func_builder=q_func_builder,
                                            q_solver_builder=q_solver_builder,
                                            replay_buffer_builder=replay_buffer_builder,
                                            explorer_builder=explorer_builder)

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.DDQNQTrainerConfig(num_steps=self._config.num_steps,
                                                                q_loss_scalar=self._config.q_loss_scalar,
                                                                reduction_method='sum',
                                                                grad_clip=self._config.grad_clip,
                                                                unroll_steps=self._config.unroll_steps,
                                                                burn_in_steps=self._config.burn_in_steps,
                                                                reset_on_terminal=self._config.reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.DDQNQTrainer(train_function=self._q,
                                                              solvers={self._q.scope_name: self._q_solver},
                                                              target_function=self._target_q,
                                                              env_info=self._env_info,
                                                              config=trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer

    def _random_action_selector(self, s, *, begin_of_episode=False):
        return self._env_info.action_space.sample(), {}

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return env_info.is_continuous_action_env()
