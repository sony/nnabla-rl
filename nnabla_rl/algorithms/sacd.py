# Copyright 2022,2023 Sony Group Corporation.
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
import numpy as np

import nnabla as nn
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import eval_api
from nnabla_rl.algorithms.common_utils import _InfluenceMetricsEvaluator
from nnabla_rl.algorithms.sac import (SAC, DefaultExplorerBuilder, DefaultPolicyBuilder, DefaultReplayBufferBuilder,
                                      DefaultSolverBuilder, SACConfig)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import QFunction, SACDQFunction, StochasticPolicy
from nnabla_rl.models.q_function import FactoredContinuousQFunction
from nnabla_rl.utils import context
from nnabla_rl.utils.misc import sync_model


@dataclass
class SACDConfig(SACConfig):
    """SACDConfig List of configurations for SACD algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0003.
        batch_size(int): training batch size. Defaults to 256.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        environment_steps (int): Number of steps to interact with the environment on each iteration. Defaults to 1.
        gradient_steps (int): Number of parameter updates to perform on each iteration. Defaults to 1.
        target_entropy (float, optional): Target entropy value. Defaults to None.
        initial_temperature (float, optional): Initial value of temperature parameter. Defaults to None.
        fix_temperature (bool): If true the temperature parameter will not be trained. Defaults to False.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        num_steps (int): number of steps for N-step Q targets. Defaults to 1.
        actor_unroll_steps (int): Number of steps to unroll actor's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        actor_burn_in_steps (int): Number of burn-in steps to initiaze actor's recurrent layer states during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        actor_reset_rnn_on_terminal (bool): Reset actor's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        critic_unroll_steps (int): Number of steps to unroll critic's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        critic_burn_in_steps (int): Number of burn-in steps to initiaze critic's recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        critic_reset_rnn_on_terminal (bool): Reset critic's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        reward_dimension (int): Number of reward components to learn.
    """

    reward_dimension: int = 1

    def __post_init__(self):
        super().__post_init__()
        self._assert_positive(self.reward_dimension, 'reward_dimension')


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: SACDConfig,
                    **kwargs) -> QFunction:
        # increment reward dimension to accomodate entropy bonus
        return SACDQFunction(scope_name, algorithm_config.reward_dimension + 1)


class SACD(SAC):
    """Soft Actor-Critic Decomposition (SAC-D) algorithm implementation.

    This class implements the factored version of Soft Actor Critic (SAC) algorithm
    proposed by J. MacGlashan, et al. in the paper: "Value Function Decomposition for Iterative
    Design of Reinforcement Learning Agents"
    For detail see: https://arxiv.org/abs/2206.13901

    This algorithm trains factored Q-function to preserve factored reward information.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`SACDConfig <nnabla_rl.algorithms.sacd.SACDConfig>`): configuration of the SACD algorithm
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of q function solvers
        policy_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of policy solvers
        temperature_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of temperature solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: SACDConfig
    _influence_metrics_evaluator: _InfluenceMetricsEvaluator

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: SACDConfig = SACDConfig(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 temperature_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(SACD, self).__init__(
            env_or_env_info=env_or_env_info,
            config=config,
            q_function_builder=q_function_builder,
            q_solver_builder=q_solver_builder,
            policy_builder=policy_builder,
            policy_solver_builder=policy_solver_builder,
            temperature_solver_builder=temperature_solver_builder,
            replay_buffer_builder=replay_buffer_builder,
            explorer_builder=explorer_builder,
        )
        assert isinstance(self._train_q_functions[0], FactoredContinuousQFunction)
        self._influence_metrics_evaluator = _InfluenceMetricsEvaluator(self._env_info, self._train_q_functions[0])

    def _setup_q_function_training(self, env_or_buffer):
        # training input/loss variables
        q_function_trainer_config = MT.q_value_trainers.SoftQDTrainerConfig(
            reduction_method='mean',
            grad_clip=None,
            num_steps=self._config.num_steps,
            unroll_steps=self._config.critic_unroll_steps,
            burn_in_steps=self._config.critic_burn_in_steps,
            reset_on_terminal=self._config.critic_reset_rnn_on_terminal,
            reward_dimension=self._config.reward_dimension)

        q_function_trainer = MT.q_value_trainers.SoftQDTrainer(
            train_functions=self._train_q_functions,
            solvers=self._train_q_solvers,
            target_functions=self._target_q_functions,
            target_policy=self._pi,
            temperature=self._policy_trainer.get_temperature(),
            env_info=self._env_info,
            config=q_function_trainer_config)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q)
        return q_function_trainer

    @eval_api
    def compute_influence_metrics(self,
                                  state: np.ndarray,
                                  action: np.ndarray,
                                  *,
                                  begin_of_episode: bool = False) -> np.ndarray:
        """Compute relative influence metrics.

        The influence metrics represent how much each reward component contributes to an agent's decisions.
        For detail see: https://arxiv.org/abs/2206.13901

        Args:
            state (np.ndarray): state to compute the influence metrics.
            action (np.ndarray): action to compute the influence metrics.
            begin_of_episode (bool): Used for rnn state resetting. This flag informs the beginning of episode.

        Returns:
            np.ndarray: Relative influence metrics for each given state and action.
        """
        # TODO: standardize API
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            influence, _ = self._influence_metrics_evaluator(state, action, begin_of_episode=begin_of_episode)
            return influence
