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
from typing import Any, Dict, Union

import gym

import nnabla as nn
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.dqn import (DQN, DefaultQFunctionBuilder, DefaultReplayBufferBuilder, DefaultSolverBuilder,
                                      DQNConfig)
from nnabla_rl.builders import ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer
from nnabla_rl.models import QFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.misc import sync_model


@dataclass
class DDQNConfig(DQNConfig):
    """
    List of configurations for Double DQN (DDQN) algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00025.
        batch_size (int): training batch size. Defaults to 32.
        num_steps (int): number of steps for N-step Q targets. Defaults to 1.
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
        test_epsilon (float): the epsilon value on testing. Defaults to 0.05.
        grad_clip (Optional[Tuple[float, float]]): Clip the gradient of final layer. Defaults to (-1.0, 1.0).
    """
    pass


class DDQN(DQN):
    '''Double DQN algorithm.

    This class implements the Deep Q-Network with double q-learning (DDQN) algorithm
    proposed by H. van Hasselt, et al. in the paper: "Deep Reinforcement Learning with Double Q-learning"
    For details see: https://arxiv.org/abs/1509.06461

    Note that default solver used in this implementation is RMSPropGraves as in the original paper.
    However, in practical applications, we recommend using Adam as the optimizer of DDQN.
    You can replace the solver by implementing a (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) and
    pass the solver on DDQN class instantiation.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DDQNConfig <nnabla_rl.algorithms.double_dqn.DDQNConfig>`):
            the parameter for DDQN training
        q_func_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of q function model
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of q function solver
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DDQNConfig
    _q: QFunction
    _q_solver: nn.solver.Solver
    _target_q: QFunction
    _replay_buffer: ReplayBuffer
    _environment_explorer: EnvironmentExplorer
    _q_function_trainer: ModelTrainer
    _eval_state_var: nn.Variable
    _a_greedy: nn.Variable

    _q_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DDQNConfig = DDQNConfig(),
                 q_func_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(DDQN, self).__init__(env_or_env_info=env_or_env_info,
                                   config=config,
                                   q_func_builder=q_func_builder,
                                   q_solver_builder=q_solver_builder,
                                   replay_buffer_builder=replay_buffer_builder)

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.DDQNQTrainerConfig(num_steps=self._config.num_steps,
                                                                reduction_method='sum',
                                                                grad_clip=self._config.grad_clip)

        q_function_trainer = MT.q_value_trainers.DDQNQTrainer(train_function=self._q,
                                                              solvers={self._q.scope_name: self._q_solver},
                                                              target_function=self._target_q,
                                                              env_info=self._env_info,
                                                              config=trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer
