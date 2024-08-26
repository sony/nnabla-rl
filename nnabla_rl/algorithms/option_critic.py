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

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import (
    _EpsilonGreedyOptionSelector,
    _GreedyOptionSelector,
    _RandomOptionSelector,
    _StochasticIntraPolicyActionSelector,
)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import (
    AtariOptionCriticIntraPolicy,
    AtariOptionCriticOptionVFunction,
    AtariOptionCriticTerminationFunction,
    OptionCriticSharedFunctionHead,
    OptionValueFunction,
    StochasticIntraPolicy,
    StochasticTerminationFunction,
)
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class OptionCriticConfig(AlgorithmConfig):
    """List of configurations for Option Critic Architecture algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        intra_policy_learning_rate (float): learning rate which is set to intra policy solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00025.
        termination_function_learning_rate (float): learning rate which is set to termination function solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00025.
        option_v_function__learning_rate (float): learning rate which is set to option value function sulver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00025.
        option_v_batch_size (int): training batch size of option value function. Defaults to 32.
        termination_function_batch_size (int): training batch size of termination function function. Defaults to 1.
        intra_policy_batch_size (int): training batch size of intra policy. Defaults to 1.
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
        advantage_offset (float): advantage offset value for termination function learning. Defaults to 0.01.
        entropy_regularizer_coefficient (float): scalar of entropy regularization term of intra policy learning. \
            Defaults to 0.01.
        use_baseline (bool): If True, subtracting the baseline value from the q value in intra policy learning. \
            Defaults to True.
        num_options (int): number of options. Defaults to 8.
        option_v_loss_reduction_method (str): The reduction method for option v function loss. Defaults to 'sum'.
        intra_policy_loss_reduction_method (str): The reduction method for intra policy loss. Defaults to 'mean'.
        termination_function_loss_reduction_method (str): The reduction method for termination function loss. \
            Defaults to 'mean'.
        deterministic_termination_in_eval (bool): If true, terminates deterministically at evalution. Defaults to False.
        deterministic_intra_action_in_eval (bool): If true, act deterministically at evalution. Defaults to False.
    """

    gamma: float = 0.99

    intra_policy_learning_rate: float = 2.5e-4
    termination_function_learning_rate: float = 2.5e-4
    option_v_function_learning_rate: float = 2.5e-4

    option_v_batch_size: int = 32
    termination_function_batch_size: int = 1
    intra_policy_batch_size: int = 1

    learner_update_frequency: float = 4
    target_update_frequency: float = 10000

    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000

    max_option_explore_steps: int = 1000000
    initial_option_epsilon: float = 1.0
    final_option_epsilon: float = 0.1
    test_option_epsilon: float = 0.05

    advantage_offset: float = 0.01
    entropy_regularizer_coefficient: float = 0.01
    use_baseline: bool = True
    num_options: int = 8

    option_v_loss_reduction_method: str = "sum"
    intra_policy_loss_reduction_method: str = "mean"
    termination_function_loss_reduction_method: str = "mean"

    deterministic_termination_in_eval: bool = False
    deterministic_intra_action_in_eval: bool = False

    def __post_init__(self):
        """__post_init__

        Check set values are in valid range.
        """
        self._assert_between(self.gamma, 0.0, 1.0, "gamma")

        self._assert_positive(self.intra_policy_learning_rate, "intra_policy_learning_rate")
        self._assert_positive(self.option_v_function_learning_rate, "option_v_function_learning_rate")
        self._assert_positive(self.termination_function_learning_rate, "termination_function_learning_rate")

        self._assert_positive(self.option_v_batch_size, "option_v_batch_size")
        self._assert_positive(self.termination_function_batch_size, "termination_function_batch_size")
        self._assert_positive(self.intra_policy_batch_size, "intra_policy_batch_size")

        self._assert_positive(self.learner_update_frequency, "learner_update_frequency")
        self._assert_positive(self.target_update_frequency, "target_update_frequency")

        self._assert_positive(self.start_timesteps, "start_timesteps")
        self._assert_positive(self.replay_buffer_size, "replay_buffer_size")
        self._assert_smaller_than(self.start_timesteps, self.replay_buffer_size, "start_timesteps")

        self._assert_between(self.initial_option_epsilon, 0.0, 1.0, "initial_option_epsilon")
        self._assert_between(self.final_option_epsilon, 0.0, 1.0, "final_option_epsilon")
        self._assert_between(self.test_option_epsilon, 0.0, 1.0, "test_option_epsilon")
        self._assert_positive(self.max_option_explore_steps, "max_option_explore_steps")

        self._assert_positive(self.num_options, "num_options")
        self._assert_positive_or_zero(self.advantage_offset, "advantage_offset")
        self._assert_positive_or_zero(self.entropy_regularizer_coefficient, "entropy_regularizer_coefficient")

        self._assert_one_of(self.option_v_loss_reduction_method, ["sum", "mean"], "option_v_loss_reduction_method")
        self._assert_one_of(
            self.intra_policy_loss_reduction_method, ["sum", "mean"], "intra_policy_loss_reduction_method"
        )
        self._assert_one_of(
            self.termination_function_loss_reduction_method,
            ["sum", "mean"],
            "termination_function_loss_reduction_method",
        )


class DefaultOptionValueFunctionBuilder(ModelBuilder[OptionValueFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: OptionCriticConfig,
        **kwargs,
    ) -> OptionValueFunction:
        # scope name is same as that of termination function and intra policy
        # -> parameter is shared across models automatically
        _shared_function_head = OptionCriticSharedFunctionHead(
            scope_name="shared", state_shape=env_info.state_shape, action_dim=env_info.action_dim
        )
        return AtariOptionCriticOptionVFunction(
            scope_name="shared",  # shared feature function should be updated via option value function loss only
            head=_shared_function_head,
            num_options=algorithm_config.num_options,
        )


class DefaultIntraPolicyBuilder(ModelBuilder[StochasticIntraPolicy]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: OptionCriticConfig,
        **kwargs,
    ) -> StochasticIntraPolicy:
        assert scope_name != "shared"
        # scope name is same as that of option v function and termination function
        # -> parameter is shared across models automatically
        _shared_function_head = OptionCriticSharedFunctionHead(
            scope_name="shared", state_shape=env_info.state_shape, action_dim=env_info.action_dim
        )
        return AtariOptionCriticIntraPolicy(
            scope_name=scope_name,
            head=_shared_function_head,
            num_options=algorithm_config.num_options,
            action_dim=env_info.action_dim,
        )


class DefaultTerminationFunctionBuilder(ModelBuilder[StochasticTerminationFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: OptionCriticConfig,
        **kwargs,
    ) -> StochasticTerminationFunction:
        assert scope_name != "shared"
        # scope name is same as that of option v function and intra policy
        # -> parameter is shared across models automatically
        _shared_function_head = OptionCriticSharedFunctionHead(
            scope_name="shared", state_shape=env_info.state_shape, action_dim=env_info.action_dim
        )
        return AtariOptionCriticTerminationFunction(
            scope_name=scope_name, head=_shared_function_head, num_options=algorithm_config.num_options
        )


class DefaultIntraPolicySolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: OptionCriticConfig, **kwargs
    ) -> nn.solver.Solver:
        solver = NS.Sgd(lr=algorithm_config.intra_policy_learning_rate)
        return solver


class DefaultTerminationFunctionSolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: OptionCriticConfig, **kwargs
    ) -> nn.solver.Solver:
        solver = NS.Sgd(lr=algorithm_config.termination_function_learning_rate)
        return solver


class DefaultOptionVFunctionSolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: OptionCriticConfig, **kwargs
    ) -> nn.solver.Solver:
        # this decay is equivalent to 'gradient momentum' and 'squared gradient momentum' of the nature paper
        decay: float = 0.95
        momentum: float = 0.0
        min_squared_gradient: float = 0.01
        solver = NS.RMSpropGraves(
            lr=algorithm_config.option_v_function_learning_rate,
            decay=decay,
            momentum=momentum,
            eps=min_squared_gradient,
        )
        return solver


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: OptionCriticConfig, **kwargs
    ) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(  # type: ignore[override]
        self,
        env_info: EnvironmentInfo,
        algorithm_config: OptionCriticConfig,
        algorithm: "OptionCritic",
        **kwargs,
    ) -> EnvironmentExplorer:
        explorer_config = EE.LinearDecayEpsilonGreedyOptionExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            timelimit_as_terminal=True,
            initial_step_num=algorithm.iteration_num,
            initial_option_epsilon=algorithm_config.initial_option_epsilon,
            final_option_epsilon=algorithm_config.final_option_epsilon,
            max_option_explore_steps=algorithm_config.max_option_explore_steps,
            num_options=algorithm_config.num_options,
            append_explorer_info=True,
        )
        explorer = EE.LinearDecayEpsilonGreedyOptionExplorer(
            env_info=env_info,
            config=explorer_config,
            random_option_selector=algorithm._exploration_random_option_selector,
            greedy_option_selector=algorithm._exploration_greedy_option_selector,
            intra_action_selector=algorithm._intra_action_selector,
        )
        return explorer


class OptionCritic(Algorithm):
    """Option Critic algorithm.

    This class implements the Option Critic Architecture algorithm
    proposed by Pierre-Luc Bacon, et al. in the paper: "The Option-Critic Architecture"
    For details see: https://arxiv.org/abs/1609.05140

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`OptionCriticConfig <nnabla_rl.algorithms.option_critic.OptionCriticConfig>`):\
            configuration of Option Critic algorithm
        option_v_func_builder (:py:class:`ModelBuilder[OptionValueFunction] \
            <nnabla_rl.builders.ModelBuilder>`): buider of option value function model
        option_v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for option value function solver
        intra_policy_builder (:py:class:`ModelBuilder[IntraPolicy] \
            <nnabla_rl.builders.ModelBuilder>`): buider of intra policy function model
        intra_policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for option value function solver
        termination_function_builder (:py:class:`ModelBuilder[TerminationFunction] \
            <nnabla_rl.builders.ModelBuilder>`): buider of termination function model
        termination_function_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for termination function solver
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: OptionCriticConfig

    _option_v_function: OptionValueFunction
    _option_v_function_solver: nn.solver.Solver
    _target_option_v_function: OptionValueFunction

    _intra_policy: StochasticIntraPolicy
    _intra_policy_solver: nn.solver.Solver

    _termination_function: StochasticTerminationFunction
    _termination_function_solver: nn.solver.Solver

    _option_v_replay_buffer: ReplayBuffer
    _explorer_builder: ExplorerBuilder
    _environment_explorer: EnvironmentExplorer

    _option_v_function_trainer: ModelTrainer
    _option_v_function_trainer_state: Dict[str, Any]

    _termination_function_trainer: ModelTrainer
    _termination_function_trainer_state: Dict[str, Any]

    _intra_policy_trainer: ModelTrainer
    _intra_policy_trainer_state: Dict[str, Any]

    _evaluation_greedy_option_actor: _GreedyOptionSelector
    _exploration_greedy_option_actor: _GreedyOptionSelector
    _evaluation_random_option_actor: _RandomOptionSelector
    _exploration_random_option_actor: _RandomOptionSelector
    _intra_action_actor: _StochasticIntraPolicyActionSelector
    _evaluation_option_actor: _EpsilonGreedyOptionSelector

    def __init__(
        self,
        env_or_env_info: Union[gym.Env, EnvironmentInfo],
        config: OptionCriticConfig = OptionCriticConfig(),
        option_v_func_builder: ModelBuilder[OptionValueFunction] = DefaultOptionValueFunctionBuilder(),
        option_v_solver_builder: SolverBuilder = DefaultOptionVFunctionSolverBuilder(),
        intra_policy_builder: ModelBuilder[StochasticIntraPolicy] = DefaultIntraPolicyBuilder(),
        intra_policy_solver_builder: SolverBuilder = DefaultIntraPolicySolverBuilder(),
        termination_func_builder: ModelBuilder[StochasticTerminationFunction] = DefaultTerminationFunctionBuilder(),
        termination_solver_builder: SolverBuilder = DefaultTerminationFunctionSolverBuilder(),
        replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
        explorer_builder: ExplorerBuilder = DefaultExplorerBuilder(),
    ):
        super(OptionCritic, self).__init__(env_or_env_info, config=config)

        self._explorer_builder = explorer_builder

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._option_v_function = option_v_func_builder(
                scope_name="shared", env_info=self._env_info, algorithm_config=self._config
            )
            self._option_v_function_solver = option_v_solver_builder(
                env_info=self._env_info, algorithm_config=self._config
            )
            self._target_option_v_function = self._option_v_function.deepcopy(
                "target_" + self._option_v_function.scope_name
            )

            self._termination_function = termination_func_builder(
                scope_name="termination_func", env_info=self._env_info, algorithm_config=self._config
            )
            self._termination_function_solver = termination_solver_builder(
                env_info=self._env_info, algorithm_config=self._config
            )

            self._intra_policy = intra_policy_builder(
                scope_name="intra_policy", env_info=self._env_info, algorithm_config=self._config
            )
            self._intra_policy_solver = intra_policy_solver_builder(
                env_info=self._env_info, algorithm_config=self._config
            )

            self._option_v_replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)
            self._termination_function_replay_buffer = ReplayBuffer(
                capacity=self._config.termination_function_batch_size
            )
            self._intra_policy_replay_buffer = ReplayBuffer(capacity=self._config.intra_policy_batch_size)
            self._environment_explorer = explorer_builder(
                env_info=self._env_info, algorithm_config=self._config, algorithm=self
            )

        self._evaluation_greedy_option_actor = _GreedyOptionSelector(
            self._config.num_options,
            self._env_info,
            self._option_v_function,
            self._termination_function,
            deterministic_termination=self._config.deterministic_termination_in_eval,
        )
        self._exploration_greedy_option_actor = _GreedyOptionSelector(
            self._config.num_options, self._env_info, self._option_v_function, self._termination_function
        )
        self._evaluation_random_option_actor = _RandomOptionSelector(
            self._config.num_options,
            self._env_info,
            self._termination_function,
            deterministic_termination=self._config.deterministic_termination_in_eval,
        )
        self._exploration_random_option_actor = _RandomOptionSelector(
            self._config.num_options,
            self._env_info,
            self._termination_function,
        )
        self._intra_action_actor = _StochasticIntraPolicyActionSelector(
            self._env_info, deterministic=self._config.deterministic_intra_action_in_eval, policy=self._intra_policy
        )

        self._evaluation_option_actor = _EpsilonGreedyOptionSelector(
            greedy_option_selector=self._evaluation_greedy_option_selector,
            random_option_selector=self._evaluation_random_option_selector,
            epsilon=self._config.test_option_epsilon,
        )

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            option, _ = self._evaluation_option_actor(state, begin_of_episode=begin_of_episode)
            action, _ = self._intra_action_selector(state, option, begin_of_episode=begin_of_episode)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._option_v_function_trainer = self._setup_option_v_function_training(env_or_buffer)
        self._intra_policy_function_trainer = self._setup_intra_policy_training(env_or_buffer)
        self._termination_function_trainer = self._setup_termination_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        return self._explorer_builder(self._env_info, self._config, self)

    def _setup_option_v_function_training(self, env_or_buffer):
        trainer_config = MT.option_value_trainers.OptionCriticOptionValueTrainerConfig(
            reduction_method=self._config.option_v_loss_reduction_method
        )
        option_v_function_trainer = MT.option_value_trainers.OptionCriticOptionValueTrainer(
            train_functions=self._option_v_function,
            solvers={self._option_v_function.scope_name: self._option_v_function_solver},
            target_function=self._target_option_v_function,
            env_info=self._env_info,
            termination_functions=self._termination_function,
            config=trainer_config,
        )
        sync_model(self._option_v_function, self._target_option_v_function)
        return option_v_function_trainer

    def _setup_intra_policy_training(self, env_or_buffer):
        trainer_config = MT.intra_policy_trainers.OptionCriticIntraPolicyTrainerConfig(
            entropy_coefficient=self._config.entropy_regularizer_coefficient,
            reduction_method=self._config.intra_policy_loss_reduction_method,
        )
        intra_policy_trainer = MT.intra_policy_trainers.OptionCriticIntraPolicyTrainer(
            models=self._intra_policy,
            solvers={self._intra_policy.scope_name: self._intra_policy_solver},
            env_info=self._env_info,
            termination_functions=self._termination_function,
            target_option_v_function=self._target_option_v_function,
            option_v_functions=self._option_v_function,
            config=trainer_config,
        )
        return intra_policy_trainer

    def _setup_termination_function_training(self, env_or_buffer):
        trainer_config = MT.termination_trainers.OptionCriticTerminationFunctionTrainerConfig(
            advantage_offset=self._config.advantage_offset,
            reduction_method=self._config.termination_function_loss_reduction_method,
        )
        termination_function_trainer = MT.termination_trainers.OptionCriticTerminationFunctionTrainer(
            models=self._termination_function,
            solvers={self._termination_function.scope_name: self._termination_function_solver},
            env_info=self._env_info,
            option_v_functions=self._option_v_function,
            config=trainer_config,
        )
        return termination_function_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)

        for e in experiences:
            s, a, r, non_terminal, n_s, info = e
            assert "option" in info
            self._option_v_replay_buffer.append((s, a, r, non_terminal, n_s, info["option"]))

            if self._config.start_timesteps < self.iteration_num:
                self._intra_policy_replay_buffer.append((s, a, r, non_terminal, n_s, info["option"]))
                self._termination_function_replay_buffer.append((s, a, r, non_terminal, n_s, info["option"]))

        if self._config.start_timesteps < self.iteration_num:

            if len(self._intra_policy_replay_buffer) >= self._config.intra_policy_batch_size:
                self._intra_policy_training(self._intra_policy_replay_buffer)
                # Clear buffer after training
                self._intra_policy_replay_buffer = ReplayBuffer(capacity=self._config.intra_policy_batch_size)

            if len(self._termination_function_replay_buffer) >= self._config.intra_policy_batch_size:
                self._termination_training(self._termination_function_replay_buffer)
                # Clear buffer after training
                self._termination_function_replay_buffer = ReplayBuffer(
                    capacity=self._config.termination_function_batch_size
                )

            if self.iteration_num % self._config.learner_update_frequency == 0:
                # off-policy training
                self._option_v_training(self._option_v_replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _evaluation_greedy_option_selector(self, s, option, *, begin_of_episode=False):
        return self._evaluation_greedy_option_actor(s, option, begin_of_episode=begin_of_episode)

    def _evaluation_random_option_selector(self, s, option, *, begin_of_episode=False):
        return self._evaluation_random_option_actor(s, option, begin_of_episode=begin_of_episode)

    def _exploration_greedy_option_selector(self, s, option, *, begin_of_episode=False):
        return self._exploration_greedy_option_actor(s, option, begin_of_episode=begin_of_episode)

    def _exploration_random_option_selector(self, s, option, *, begin_of_episode=False):
        return self._exploration_random_option_actor(s, option, begin_of_episode=begin_of_episode)

    def _intra_action_selector(self, s, option, *, begin_of_episode=False):
        return self._intra_action_actor(s, option, begin_of_episode=begin_of_episode)

    def _option_v_training(self, replay_buffer: ReplayBuffer):
        experiences, _ = replay_buffer.sample(self._config.option_v_batch_size)
        experiences = cast(Tuple[Experience], experiences)
        s, a, r, non_terminal, s_next, option = marshal_experiences(experiences)

        batch = TrainingBatch(
            batch_size=self._config.option_v_batch_size,
            s_current=s,
            a_current=a,
            gamma=self._config.gamma,
            reward=r,
            non_terminal=non_terminal,
            s_next=s_next,
            extra={"option": option},
        )

        self._option_v_function_trainer_state = self._option_v_function_trainer.train(batch)
        if self.iteration_num % self._config.target_update_frequency == 0:
            sync_model(self._option_v_function, self._target_option_v_function)

    def _termination_training(self, replay_buffer: ReplayBuffer):
        experiences, _ = replay_buffer.sample_indices(np.arange(self._config.termination_function_batch_size).tolist())
        experiences = cast(Tuple[Experience], experiences)
        s, a, r, non_terminal, s_next, option = marshal_experiences(experiences)

        batch = TrainingBatch(
            batch_size=len(experiences),
            s_current=s,
            a_current=a,
            gamma=self._config.gamma,
            reward=r,
            non_terminal=non_terminal,
            s_next=s_next,
            extra={"option": option},
        )
        self._termination_function_trainer_state = self._termination_function_trainer.train(batch)

    def _intra_policy_training(self, replay_buffer: ReplayBuffer):
        experiences, _ = replay_buffer.sample_indices(np.arange(self._config.intra_policy_batch_size).tolist())
        experiences = cast(Tuple[Experience], experiences)
        s, a, r, non_terminal, s_next, option = marshal_experiences(experiences)

        batch = TrainingBatch(
            batch_size=len(experiences),
            s_current=s,
            a_current=a,
            gamma=self._config.gamma,
            reward=r,
            non_terminal=non_terminal,
            s_next=s_next,
            extra={"option": option},
        )
        self._intra_policy_trainer_state = self._intra_policy_function_trainer.train(batch)

    def _models(self):
        models = {}
        models[self._option_v_function.scope_name] = self._option_v_function
        models[self._termination_function.scope_name] = self._termination_function
        models[self._intra_policy.scope_name] = self._intra_policy
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._option_v_function.scope_name] = self._option_v_function_solver
        solvers[self._termination_function.scope_name] = self._termination_function_solver
        solvers[self._intra_policy.scope_name] = self._intra_policy_solver
        return solvers

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = (
            EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) else env_or_env_info
        )
        return not env_info.is_continuous_action_env() and not env_info.is_tuple_action_env()

    @classmethod
    def is_rnn_supported(self):
        return False

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(OptionCritic, self).latest_iteration_state
        if hasattr(self, "_option_v_function_trainer_state"):
            latest_iteration_state["scalar"].update(
                {"option_v_loss": float(self._option_v_function_trainer_state["option_v_loss"])}
            )
        if hasattr(self, "_termination_function_trainer_state"):
            latest_iteration_state["scalar"].update(
                {"termination_loss": float(self._termination_function_trainer_state["termination_loss"])}
            )
        if hasattr(self, "_intra_policy_trainer_state"):
            latest_iteration_state["scalar"].update(
                {"intra_pi_loss": float(self._intra_policy_trainer_state["intra_pi_loss"])}
            )
        return latest_iteration_state

    @property
    def trainers(self):
        return {
            "option_v_function": self._option_v_function_trainer,
            "intra_policy": self._intra_policy_trainer,
            "termination_function": self._termination_function_trainer,
        }
