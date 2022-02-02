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
from typing import Optional, Union

import gym

import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms import ICML2018SAC, ICML2018SACConfig
from nnabla_rl.algorithms.icml2018_sac import (DefaultExplorerBuilder, DefaultPolicyBuilder, DefaultQFunctionBuilder,
                                               DefaultReplayBufferBuilder, DefaultSolverBuilder,
                                               DefaultVFunctionBuilder)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import QFunction, StochasticPolicy, VFunction
from nnabla_rl.utils.misc import sync_model


@dataclass
class MMESACConfig(ICML2018SACConfig):
    '''MMESACConfig
    List of configurations for MMESAC algorithm.

    Args:
        alpha_pi (Optional[float]): If None, will use reward_scalar to scale the reward.
            Otherwise 1/alpha_pi will be used to scale the reward. Defaults to None.
        alpha_q (float): Temperature value for negative entropy term. Defaults to 1.0.
    '''
    # override configurations
    reward_scalar: float = 5.0
    alpha_pi: Optional[float] = None
    alpha_q: float = 1.0

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        super().__post_init__()
        if self.alpha_pi is not None:
            # Recompute with alpha_pi
            self.reward_scalar = 1 / self.alpha_pi


class MMESAC(ICML2018SAC):
    '''Max-Min Entropy Soft Actor-Critic (MME-SAC) algorithm.

    This class implements the Max-Min Entropy Soft Actor Critic (MME-SAC) algorithm proposed by S. Han, et al.
    in the paper: "A Max-Min Entropy Framework for Reinforcement Learning"
    For details see: https://arxiv.org/abs/2106.10517

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`MMESACConfig <nnabla_rl.algorithms.mme_sac.MMESACConfig>`):
            configuration of the MMESAC algorithm
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of v function solvers
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of q function solvers
        policy_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of policy solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: MMESACConfig

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: MMESACConfig = MMESACConfig(),
                 v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(MMESAC, self).__init__(env_or_env_info,
                                     config=config,
                                     v_function_builder=v_function_builder,
                                     v_solver_builder=v_solver_builder,
                                     q_function_builder=q_function_builder,
                                     q_solver_builder=q_solver_builder,
                                     policy_builder=policy_builder,
                                     policy_solver_builder=policy_solver_builder,
                                     replay_buffer_builder=replay_buffer_builder,
                                     explorer_builder=explorer_builder)

    def _setup_v_function_training(self, env_or_buffer):
        alpha_q = MT.policy_trainers.soft_policy_trainer.AdjustableTemperature(
            scope_name='alpha_q',
            initial_value=self._config.alpha_q)
        v_function_trainer_config = MT.v_value_trainers.MMEVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=0.5,
            unroll_steps=self._config.v_unroll_steps,
            burn_in_steps=self._config.v_burn_in_steps,
            reset_on_terminal=self._config.v_reset_rnn_on_terminal)
        v_function_trainer = MT.v_value_trainers.MMEVTrainer(
            train_functions=self._v,
            temperature=alpha_q,
            solvers={self._v.scope_name: self._v_solver},
            target_functions=self._train_q_functions,  # Set training q as target
            target_policy=self._pi,
            env_info=self._env_info,
            config=v_function_trainer_config)
        sync_model(self._v, self._target_v, 1.0)

        return v_function_trainer
