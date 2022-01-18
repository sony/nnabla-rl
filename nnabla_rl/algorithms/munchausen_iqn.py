# Copyright 2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.iqn import (IQN, DefaultExplorerBuilder, DefaultQuantileFunctionBuilder,
                                      DefaultReplayBufferBuilder, DefaultSolverBuilder, IQNConfig, risk_neutral_measure)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import StateActionQuantileFunction
from nnabla_rl.utils.misc import sync_model


@dataclass
class MunchausenIQNConfig(IQNConfig):
    """
    List of configurations for Munchausen IQN algorithm

    Args:
        entropy_temperature (float): temperature parameter of softmax policy distribution. Defaults to 0.03.
        munchausen_scaling_term (float): scalar of scaled log policy. Defaults to 0.9.
        clipping_value (float): Lower value of the logarithm of policy distribution. Defaults to -1.
    """

    # munchausen iqn training parameters
    entropy_temperature: float = 0.03
    munchausen_scaling_term: float = 0.9
    clipping_value: float = -1

    def __post_init__(self):
        '''__post_init__

        Check that set values are in valid range.

        '''
        super().__post_init__()
        self._assert_positive(self.embedding_dim, 'embedding_dim')
        self._assert_negative(self.clipping_value, 'clipping_value')


class MunchausenIQN(IQN):
    '''Munchausen-IQN algorithm implementation.

    This class implements the Munchausen-IQN (Munchausen Implicit Quantile Network) algorithm
    proposed by N. Vieillard, et al. in the paper: "Munchausen Reinforcement Learning"
    For details see: https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`MunchausenIQNConfig <nnabla_rl.algorithms.munchausen_iqn.MunchausenIQNConfig>`):
            configuration of MunchausenIQN algorithm
        risk_measure_function (Callable[[nn.Variable], nn.Variable]): risk measure function to apply to the quantiles.
        quantile_function_builder (:py:class:`ModelBuilder[StateActionQuantileFunction] \
            <nnabla_rl.builders.ModelBuilder>`): builder of state-action quantile function models
        quantile_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for state action quantile function solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: MunchausenIQNConfig

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: MunchausenIQNConfig = MunchausenIQNConfig(),
                 risk_measure_function=risk_neutral_measure,
                 quantile_function_builder: ModelBuilder[StateActionQuantileFunction]
                 = DefaultQuantileFunctionBuilder(),
                 quantile_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(MunchausenIQN, self).__init__(env_or_env_info, config=config,
                                            risk_measure_function=risk_measure_function,
                                            quantile_function_builder=quantile_function_builder,
                                            quantile_solver_builder=quantile_solver_builder,
                                            replay_buffer_builder=replay_buffer_builder,
                                            explorer_builder=explorer_builder)

    def _setup_quantile_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.MunchausenIQNQTrainerConfig(
            num_steps=self._config.num_steps,
            N=self._config.N,
            N_prime=self._config.N_prime,
            K=self._config.K,
            kappa=self._config.kappa,
            tau=self._config.entropy_temperature,
            alpha=self._config.munchausen_scaling_term,
            clip_min=self._config.clipping_value,
            clip_max=0.0,
            unroll_steps=self._config.unroll_steps,
            burn_in_steps=self._config.burn_in_steps,
            reset_on_terminal=self._config.reset_rnn_on_terminal)

        quantile_function_trainer = MT.q_value_trainers.MunchausenIQNQTrainer(
            train_functions=self._quantile_function,
            solvers={self._quantile_function.scope_name: self._quantile_function_solver},
            target_function=self._target_quantile_function,
            env_info=self._env_info,
            config=trainer_config)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        sync_model(self._quantile_function, self._target_quantile_function)

        return quantile_function_trainer
