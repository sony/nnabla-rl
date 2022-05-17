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

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.dqn import DQN, DefaultExplorerBuilder, DefaultReplayBufferBuilder, DQNConfig
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import DRQNQFunction, QFunction
from nnabla_rl.utils.misc import sync_model
from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm


@dataclass
class DRQNConfig(DQNConfig):
    """
    List of configurations for DRQN algorithm. Most of the configs are inherited from DQNConfig

    Args:
        clip_grad_norm (float): Limit the model parameter's gradient on parameter updates up to this value.
            If you implement SolverBuilder by yourself, this value will not take effect. Defaults to 10.0.
        learning_rate (float): Solver learning rate. Value overridden from DQN. Defaults to 0.1.
        replay_buffer_size (int): Replay buffer size. Value overridden from DQN. Defaults to 400000.
        unroll_steps (int): Number of steps to unroll recurrent layer during training.
            Value overridden from DQN. Defaults to 10.
        reset_rnn_on_terminal (bool): Reset recurrent internal states to zero during training if episode ends.
            Value overridden from DQN. Defaults to False.
    """

    clip_grad_norm: float = 10.0

    # Overriding some configurations from original DQNConfig
    learning_rate: float = 0.1
    replay_buffer_size: int = 400000
    unroll_steps: int = 10
    reset_rnn_on_terminal: bool = False


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DRQNConfig,
                     **kwargs) -> nn.solver.Solver:
        decay: float = 0.95
        solver = NS.Adadelta(lr=algorithm_config.learning_rate, decay=decay)
        solver = AutoClipGradByNorm(solver, algorithm_config.clip_grad_norm)
        return solver


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DRQNConfig,
                    **kwargs) -> QFunction:
        return DRQNQFunction(scope_name, env_info.action_dim)


class DRQN(DQN):
    '''DRQN algorithm.

    This class implements the Bootstrapped random update version of Deep Recurrent Q-Network (DRQN) algorithm.
    proposed by M. Hausknecht, et al. in the paper: "Deep Recurrent Q-Learning for Partially Observable MDPs"
    For details see: https://arxiv.org/pdf/1507.06527.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DRQNConfig <nnabla_rl.algorithms.drqn.DRQNConfig>`):
            the parameter for DRQN training
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
    _config: DRQNConfig

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DRQNConfig = DRQNConfig(),
                 q_func_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(DRQN, self).__init__(env_or_env_info,
                                   config=config,
                                   q_func_builder=q_func_builder,
                                   q_solver_builder=q_solver_builder,
                                   replay_buffer_builder=replay_buffer_builder,
                                   explorer_builder=explorer_builder)

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.DQNQTrainerConfig(
            num_steps=self._config.num_steps,
            reduction_method='mean',  # This parameter is different from DQN
            grad_clip=self._config.grad_clip,
            unroll_steps=self._config.unroll_steps,
            burn_in_steps=self._config.burn_in_steps,
            reset_on_terminal=self._config.reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.DQNQTrainer(
            train_functions=self._q,
            solvers={self._q.scope_name: self._q_solver},
            target_function=self._target_q,
            env_info=self._env_info,
            config=trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer
