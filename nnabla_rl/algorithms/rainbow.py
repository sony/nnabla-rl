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
import nnabla_rl.environment_explorers as EE
from nnabla_rl.algorithms.categorical_ddqn import CategoricalDDQN, CategoricalDDQNConfig
from nnabla_rl.algorithms.categorical_dqn import CategoricalDQN
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import RainbowValueDistributionFunction, ValueDistributionFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers import ProportionalPrioritizedReplayBuffer


@dataclass
class RainbowConfig(CategoricalDDQNConfig):
    '''RainbowConfig
    List of configurations for Rainbow algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00025 / 4.
        batch_size (int): training batch size. Defaults to 32.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.
            Defaults to 20000.
        replay_buffer_size (int): the capacity of replay buffer. Defaults to 1000000.
        learner_update_frequency (float): the interval of learner update. Defaults to 4.
        target_update_frequency (float): the interval of target q-function update. Defaults to 8000.
        v_min (float): lower limit of the value used in value distribution function. Defaults to -10.0.
        v_max (float): upper limit of the value used in value distribution function. Defaults to 10.0.
        num_atoms (int): the number of bins used in value distribution function. Defaults to 51.
        num_steps (int): the of steps to look ahead in n-step Q learning. Defaults to 3.
        alpha (float): priority exponent (written as omega in the rainbow paper) of prioritized buffer. Defaults to 0.5.
        beta (float): initial value of importance sampling exponent of prioritized buffer. Defaults to 0.4.
        betasteps (int): importance sampling exponent increase steps. After betasteps, exponent will get to 1.0.
            Defaults to 12500000.
        warmup_random_steps (Optional[int]): steps until this value will NOT use trained policy for exploration.
            Will explore with randomly selected action. Defaults to 0.
        no_double (bool): If true, following normal Q-learning style q value target will be used for
            categorical q value update.  :math:`r + \\gamma\\max_{a}{Q_{\\text{target}}(s_{t+1}, a)}`.
            Defaults to False.
    '''
    learning_rate: float = 0.00025 / 4
    start_timesteps: int = 20000  # 20k steps = 80k frames in Atari game
    target_update_frequency: int = 8000  # 8k steps = 32k frames in Atari game
    num_steps: int = 3
    initial_epsilon: float = 0.0  # Does not take effect by default.
    final_epsilon: float = 0.0  # Does not take effect by default.
    test_epsilon: float = 0.0
    alpha: float = 0.5
    beta: float = 0.4
    betasteps: int = 50000000 // 4  # 50M steps = 200M frames in Atari game. We diveded by 4 (learner_update_frequency).
    warmup_random_steps: int = 0
    no_double: bool = False


class DefaultValueDistFunctionBuilder(ModelBuilder[ValueDistributionFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: RainbowConfig,
                    **kwargs) -> ValueDistributionFunction:
        return RainbowValueDistributionFunction(scope_name,
                                                env_info.action_dim,
                                                algorithm_config.num_atoms,
                                                algorithm_config.v_min,
                                                algorithm_config.v_max)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: RainbowConfig,
                            **kwargs) -> ReplayBuffer:
        return ProportionalPrioritizedReplayBuffer(capacity=algorithm_config.replay_buffer_size,
                                                   alpha=algorithm_config.alpha,
                                                   beta=algorithm_config.beta,
                                                   betasteps=algorithm_config.betasteps,
                                                   error_clip=(-100, 100),
                                                   normalization_method="batch_max")


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: RainbowConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate, eps=1.5e-4)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: RainbowConfig,
                       algorithm: "Rainbow",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.RawPolicyExplorerConfig(
            warmup_random_steps=algorithm_config.warmup_random_steps,
            initial_step_num=algorithm.iteration_num
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=algorithm._greedy_action_selector,
                                        env_info=env_info,
                                        config=explorer_config)
        return explorer


class Rainbow(CategoricalDDQN):
    '''Rainbow algorithm.
    This class implements the Rainbow algorithm proposed by M. Bellemare, et al. in the paper:
    "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    For details see: https://arxiv.org/abs/1710.02298

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`RainbowConfig <nnabla_rl.algorithms.rainbow.RainbowConfig>`):
            configuration of the Rainbow algorithm
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
                 config: RainbowConfig = RainbowConfig(),
                 value_distribution_builder: ModelBuilder[ValueDistributionFunction]
                 = DefaultValueDistFunctionBuilder(),
                 value_distribution_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(Rainbow, self).__init__(env_or_env_info,
                                      config=config,
                                      value_distribution_builder=value_distribution_builder,
                                      value_distribution_solver_builder=value_distribution_solver_builder,
                                      replay_buffer_builder=replay_buffer_builder,
                                      explorer_builder=explorer_builder)

    def _setup_value_distribution_function_training(self, env_or_buffer):
        if self._config.no_double:
            return CategoricalDQN._setup_value_distribution_function_training(self, env_or_buffer)
        else:
            return CategoricalDDQN._setup_value_distribution_function_training(self, env_or_buffer)
