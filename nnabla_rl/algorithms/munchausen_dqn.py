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

from dataclasses import dataclass
from typing import Union

import gym

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.dqn import (DQN, DefaultExplorerBuilder, DefaultQFunctionBuilder, DefaultReplayBufferBuilder,
                                      DQNConfig)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import QFunction
from nnabla_rl.utils.misc import sync_model


@dataclass
class MunchausenDQNConfig(DQNConfig):
    """
    List of configurations for Munchausen DQN algorithm

    Args:
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00005.
        final_epsilon (float): the last epsilon value for ε-greedy explorer. Defaults to 0.01.
        test_epsilon (float): the epsilon value on testing. Defaults to 0.001.
        entropy_temperature (float): temperature parameter of softmax policy distribution. Defaults to 0.03.
        munchausen_scaling_term (float): scalar of scaled log policy. Defaults to 0.9.
        clipping_value (float): Lower value of the logarithm of policy distribution. Defaults to -1.
    """

    # Parameters overridden from DQN
    learning_rate: float = 0.00005
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    # munchausen dqn training parameters
    entropy_temperature: float = 0.03
    munchausen_scaling_term: float = 0.9
    clipping_value: float = -1

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        super().__post_init__()
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_negative(self.clipping_value, 'clipping_value')


class DefaultQSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: MunchausenDQNConfig,
                     **kwargs) -> nn.solvers.Solver:
        assert isinstance(algorithm_config, MunchausenDQNConfig)
        return NS.Adam(algorithm_config.learning_rate, eps=1e-2 / algorithm_config.batch_size)


class MunchausenDQN(DQN):
    '''Munchausen-DQN algorithm.

    This class implements the Munchausen-DQN (Munchausen Deep Q Network) algorithm
    proposed by N. Vieillard, et al. in the paper: "Munchausen Reinforcement Learning"
    For details see: https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`MunchausenDQNConfig <nnabla_rl.algorithms.munchausen_dqn.MunchausenDQNConfig>`):
            configuration of MunchausenDQN algorithm
        q_func_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q-function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for q-function solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: MunchausenDQNConfig

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: MunchausenDQNConfig = MunchausenDQNConfig(),
                 q_func_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultQSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(MunchausenDQN, self).__init__(env_or_env_info=env_or_env_info,
                                            config=config,
                                            q_func_builder=q_func_builder,
                                            q_solver_builder=q_solver_builder,
                                            replay_buffer_builder=replay_buffer_builder,
                                            explorer_builder=explorer_builder)

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.MunchausenDQNQTrainerConfig(
            num_steps=self._config.num_steps,
            reduction_method='mean',
            q_loss_scalar=0.5,
            grad_clip=(-1.0, 1.0),
            tau=self._config.entropy_temperature,
            alpha=self._config.munchausen_scaling_term,
            clip_min=self._config.clipping_value,
            clip_max=0.0,
            unroll_steps=self._config.unroll_steps,
            burn_in_steps=self._config.burn_in_steps,
            reset_on_terminal=self._config.reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.MunchausenDQNQTrainer(
            train_functions=self._q,
            solvers={self._q.scope_name: self._q_solver},
            target_function=self._target_q,
            env_info=self._env_info,
            config=trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer
