# Copyright 2024 Sony Group Corporation.
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

import warnings
from dataclasses import dataclass
from typing import Sequence, Union

import gym

import nnabla as nn
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.sac import (
    SAC,
    DefaultExplorerBuilder,
    DefaultPolicyBuilder,
    DefaultQFunctionBuilder,
    DefaultReplayBufferBuilder,
    DefaultSolverBuilder,
    SACConfig,
)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import Model, QFunction, StochasticPolicy
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences


@dataclass
class SRSACConfig(SACConfig):
    """SRSACConfig List of configurations for SRSAC algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0003.
        batch_size(int): training batch size. Defaults to 256.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        environment_steps (int): Number of steps to interact with the environment on each iteration. Defaults to 1.
        gradient_steps (int): Number of parameter updates to perform on each iteration. Defaults to 1. \
            Keep this value to 1 and use replay_ratio to control the number of updates in SRSAC.
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
        replay_ratio (int): Number of updates per environment step. Defaults to 1.
        reset_interval (int): Paramerters will be reset every this number of updates. Defaults to 2560000.
    """

    replay_ratio: int = 1
    reset_interval: int = 2560000  # 2.56 * 10^6

    def __post_init__(self):
        super().__post_init__()
        self._assert_positive(self.replay_ratio, "replay_ratio")
        self._assert_positive(self.reset_interval, "reset_interval")


class SRSAC(SAC):
    """Scaled-by-Resetting Soft Actor-Critic (SRSAC) algorithm implementation.

    This class implements Scaled-by-Restting Soft Actor Critic (SRSAC) algorithm proposed by P. D'Oro, et al.
    in the paper: "Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier".
    For details see: https://openreview.net/forum?id=OpC-9aBBVJe

    This algorithm periodically resets the models and optimizers' parameters for stable and efficient learning.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`SRSACConfig <nnabla_rl.algorithms.sacd.SRSACConfig>`): configuration of the SRSAC algorithm
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
    _config: SRSACConfig

    def __init__(
        self,
        env_or_env_info: Union[gym.Env, EnvironmentInfo],
        config: SRSACConfig = SRSACConfig(),
        q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
        q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
        policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        temperature_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
        explorer_builder: ExplorerBuilder = DefaultExplorerBuilder(),
    ):
        super(SRSAC, self).__init__(
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

    def _run_online_training_iteration(self, env):
        for _ in range(self._config.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._config.gradient_steps):
            if self._config.start_timesteps < self.iteration_num:
                self._run_gradient_step(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._run_gradient_step(buffer)

    def _run_gradient_step(self, replay_buffer):
        for _ in range(self._config.replay_ratio):
            self._sac_training(replay_buffer)

        num_updates = (self.iteration_num * self._config.replay_ratio) % self._config.reset_interval
        num_updates += self._config.replay_ratio
        if self._config.reset_interval <= num_updates:
            self._reset_model_parameters(self._models().values())
            self._reconstruct_training_graphs()
            self._reconstruct_actors()

    def _reset_model_parameters(self, models: Sequence[Model]):
        solvers = self._solvers()
        for model in models:
            model.clear_parameters()
            solver: nn.solvers.Solver = solvers[model.scope_name]
            solver.clear_parameters()

    def _reconstruct_training_graphs(self):
        self._temperature = self._setup_temperature_model()
        self._policy_trainer = self._setup_policy_training(env_or_buffer=None)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer=None)

    def _reconstruct_actors(self):
        self._evaluation_actor = self._setup_evaluation_actor()
        self._exploration_actor = self._setup_exploration_actor()


@dataclass
class EfficientSRSACConfig(SRSACConfig):
    """EfficientSRSACConfig List of configurations for EfficientSRSAC
    algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0003.
        batch_size(int): training batch size. Defaults to 256.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        environment_steps (int): Number of steps to interact with the environment on each iteration. Defaults to 1.
        gradient_steps (int): Number of parameter updates to perform on each iteration. Defaults to 1. \
            Keep this value to 1 and use replay_ratio to control the number of updates in SRSAC.
        target_entropy (float, optional): Target entropy value. Defaults to None.
        initial_temperature (float, optional): Initial value of temperature parameter. Defaults to None.
        fix_temperature (bool): If true the temperature parameter will not be trained. Defaults to False.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        num_steps (int): Not supported. This configuration does not take effect in the training.
        actor_unroll_steps (int): Not supported. This configuration does not take effect in the training.
        actor_burn_in_steps (int): Not supported. This configuration does not take effect in the training.
        actor_reset_rnn_on_terminal (bool): Not supported. This configuration does not take effect in the training.
        critic_unroll_steps (int): Not supported. This configuration does not take effect in the training.
        critic_burn_in_steps (int): Not supported. This configuration does not take effect in the training.
        critic_reset_rnn_on_terminal (bool): Not supported. This configuration does not take effect in the training.
        replay_ratio (int): Number of updates per environment step.
        reset_interval (int): Paramerters will be reset every this number of updates.
    """

    actor_reset_rnn_on_terminal: bool = False
    critic_reset_rnn_on_terminal: bool = False

    def __post_init__(self):
        super().__post_init__()

        def fill_warning_message(config_name, config_value, expected_value):
            return f"""{config_name} is set to {config_value}(!={expected_value})
                        but this value does not take any effect on EfficentSRSAC."""

        if 1 != self.num_steps:
            warnings.warn(fill_warning_message("num_steps", self.num_steps, 1))
        if 0 != self.actor_burn_in_steps:
            warnings.warn(fill_warning_message("actor_burn_in_steps", self.actor_burn_in_steps, 0))
        if 1 != self.actor_unroll_steps:
            warnings.warn(fill_warning_message("actor_unroll_steps", self.actor_unroll_steps, 1))
        if self.actor_reset_rnn_on_terminal:
            warnings.warn(fill_warning_message("actor_reset_rnn_on_terminal", self.actor_reset_rnn_on_terminal, False))
        if 0 != self.critic_burn_in_steps:
            warnings.warn(fill_warning_message("critic_burn_in_steps", self.critic_burn_in_steps, 0))
        if 1 != self.critic_unroll_steps:
            warnings.warn(fill_warning_message("critic_unroll_steps", self.critic_unroll_steps, 1))
        if self.critic_reset_rnn_on_terminal:
            warnings.warn(
                fill_warning_message("critic_reset_rnn_on_terminal", self.critic_reset_rnn_on_terminal, False)
            )


class EfficientSRSAC(SRSAC):
    """Efficient implementation of Scaled-by-Resetting Soft Actor-Critic
    (SRSAC) algorithm.

    This class implements a computationally efficient version of Scaled-by-Restting Soft Actor Critic (SRSAC) algorithm
    proposed by P. D'Oro, et al. in the paper: "Sample-Efficient Reinforcement Learning by Breaking
    the Replay Ratio Barrier".

    For details see: https://openreview.net/forum?id=OpC-9aBBVJe

    This implementation does not support recurrent networks. For recurrent network support use SRSAC class.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`SRSACConfig <nnabla_rl.algorithms.sacd.SRSACConfig>`): configuration of the SRSAC algorithm
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
    _config: EfficientSRSACConfig

    def __init__(
        self,
        env_or_env_info: Union[gym.Env, EnvironmentInfo],
        config: EfficientSRSACConfig = EfficientSRSACConfig(),
        q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
        q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
        policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        temperature_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
        explorer_builder: ExplorerBuilder = DefaultExplorerBuilder(),
    ):
        super().__init__(
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

    @classmethod
    def is_rnn_supported(cls):
        return False

    def _run_offline_training_iteration(self, buffer):
        self._run_gradient_step(buffer)

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._actor_critic_trainer = self._setup_actor_critic_training(env_or_buffer)

    def _setup_actor_critic_training(self, env_or_buffer):
        actor_critic_trainer_config = MT.hybrid_trainers.SRSACActorCriticTrainerConfig(
            fixed_temperature=self._config.fix_temperature,
            target_entropy=self._config.target_entropy,
            replay_ratio=self._config.replay_ratio,
            tau=self._config.tau,
        )
        actor_critic_trainer = MT.hybrid_trainers.SRSACActorCriticTrainer(
            pi=self._pi,
            pi_solvers={self._pi.scope_name: self._pi_solver},
            q_functions=self._train_q_functions,
            q_solvers=self._train_q_solvers,
            target_q_functions=self._target_q_functions,
            temperature=self._temperature,
            temperature_solver=self._temperature_solver,
            env_info=self._env_info,
            config=actor_critic_trainer_config,
        )
        return actor_critic_trainer

    def _run_gradient_step(self, replay_buffer):
        self._efficient_srsac_training(replay_buffer)

        num_updates = (self.iteration_num * self._config.replay_ratio) % self._config.reset_interval
        num_updates += self._config.replay_ratio
        if self._config.reset_interval <= num_updates:
            self._reset_model_parameters(self._models().values())
            self._reconstruct_training_graphs()
            self._reconstruct_actors()

    def _efficient_srsac_training(self, replay_buffer):
        num_steps = self._config.replay_ratio
        experiences_tuple = []
        info_tuple = []
        for _ in range(num_steps):
            experiences, info = replay_buffer.sample(self._config.batch_size)
            experiences_tuple.append(experiences)
            info_tuple.append(info)
        assert len(experiences_tuple) == num_steps

        batch = None
        for experiences, info in zip(experiences_tuple, info_tuple):
            (s, a, r, non_terminal, s_next, rnn_states_dict, *_) = marshal_experiences(experiences)
            rnn_states = rnn_states_dict["rnn_states"] if "rnn_states" in rnn_states_dict else {}
            batch = TrainingBatch(
                batch_size=self._config.batch_size,
                s_current=s,
                a_current=a,
                gamma=self._config.gamma,
                reward=r,
                non_terminal=non_terminal,
                s_next=s_next,
                weight=info["weights"],
                next_step_batch=batch,
                rnn_states=rnn_states,
            )

        self._actor_critic_trainer_state = self._actor_critic_trainer.train(batch)

        td_errors = self._actor_critic_trainer_state["td_errors"]
        replay_buffer.update_priorities(td_errors)

    def _reconstruct_training_graphs(self):
        self._temperature = self._setup_temperature_model()
        self._actor_critic_trainer = self._setup_actor_critic_training(env_or_buffer=None)

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(SAC, self).latest_iteration_state
        if hasattr(self, "_actor_critic_trainer_state"):
            latest_iteration_state["scalar"].update({"pi_loss": float(self._actor_critic_trainer_state["pi_loss"])})
        if hasattr(self, "_actor_critic_trainer_state"):
            latest_iteration_state["scalar"].update({"q_loss": float(self._actor_critic_trainer_state["q_loss"])})
            latest_iteration_state["histogram"].update(
                {"td_errors": self._actor_critic_trainer_state["td_errors"].flatten()}
            )
        return latest_iteration_state
