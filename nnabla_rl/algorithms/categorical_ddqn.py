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
from nnabla_rl.algorithms.categorical_dqn import (CategoricalDQN, CategoricalDQNConfig, DefaultExplorerBuilder,
                                                  DefaultReplayBufferBuilder, DefaultSolverBuilder,
                                                  DefaultValueDistFunctionBuilder)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import ValueDistributionFunction
from nnabla_rl.utils.misc import sync_model


@dataclass
class CategoricalDDQNConfig(CategoricalDQNConfig):
    pass


class CategoricalDDQN(CategoricalDQN):
    '''Categorical Double DQN algorithm.

    This class implements the Categorical Double DQN algorithm introduced by M. Bellemare, et al.
    in the paper: "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    For details see: https://arxiv.org/abs/1710.02298.
    The difference between Categorical DQN and this algorithm is the update target of q-value.
    This algorithm uses following double DQN style q-value target for Categorical Q value update.
    :math:`r + \\gamma Q_{\\text{target}}(s_{t+1}, \\arg\\max_{a}{Q(s_{t+1}, a)})`.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`CategoricalDDQNConfig <nnabla_rl.algorithms.categorical_ddqn.CategoricalDDQNConfig>`):
            configuration of the CategoricalDDQN algorithm
        value_distribution_builder (:py:class:`ModelBuilder[ValueDistributionFunctionFunction] \
            <nnabla_rl.builders.ModelBuilder>`): builder of value distribution function models
        value_distribution_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of value distribution function solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: CategoricalDQNConfig = CategoricalDDQNConfig(),
                 value_distribution_builder: ModelBuilder[ValueDistributionFunction]
                 = DefaultValueDistFunctionBuilder(),
                 value_distribution_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(CategoricalDDQN, self).__init__(env_or_env_info,
                                              config=config,
                                              value_distribution_builder=value_distribution_builder,
                                              value_distribution_solver_builder=value_distribution_solver_builder,
                                              replay_buffer_builder=replay_buffer_builder,
                                              explorer_builder=explorer_builder)

    def _setup_value_distribution_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.CategoricalDDQNQTrainerConfig(
            num_steps=self._config.num_steps,
            v_min=self._config.v_min,
            v_max=self._config.v_max,
            num_atoms=self._config.num_atoms,
            reduction_method=self._config.loss_reduction_method,
            unroll_steps=self._config.unroll_steps,
            burn_in_steps=self._config.burn_in_steps,
            reset_on_terminal=self._config.reset_rnn_on_terminal)

        model_trainer = MT.q_value_trainers.CategoricalDDQNQTrainer(
            train_function=self._atom_p,
            solvers={self._atom_p.scope_name: self._atom_p_solver},
            target_function=self._target_atom_p,
            env_info=self._env_info,
            config=trainer_config)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        sync_model(self._atom_p, self._target_atom_p)
        return model_trainer
