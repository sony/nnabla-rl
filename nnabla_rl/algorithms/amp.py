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

import copy
import multiprocessing as mp
import os
import threading as th
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import (
    _get_shape,
    _StatePreprocessedRewardFunction,
    _StatePreprocessedStochasticPolicy,
    _StatePreprocessedVFunction,
    _StochasticPolicyActionSelector,
)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, PreprocessorBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.amp_env import AMPEnv, AMPGoalEnv, TaskResult
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.functions import compute_std, unnormalize
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import (
    AMPDiscriminator,
    AMPGatedPolicy,
    AMPGatedVFunction,
    AMPPolicy,
    AMPVFunction,
    Model,
    RewardFunction,
    StochasticPolicy,
    VFunction,
)
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.random import drng
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.typing import Experience
from nnabla_rl.utils import context
from nnabla_rl.utils.data import (
    add_batch_dimension,
    compute_std_ndarray,
    marshal_experiences,
    normalize_ndarray,
    set_data_to_variable,
    unnormalize_ndarray,
)
from nnabla_rl.utils.misc import create_variable
from nnabla_rl.utils.multiprocess import (
    copy_mp_arrays_to_params,
    copy_params_to_mp_arrays,
    mp_array_from_np_array,
    mp_to_np_array,
    new_mp_arrays_from_params,
    np_to_mp_array,
)
from nnabla_rl.utils.reproductions import set_global_seed


@dataclass
class AMPConfig(AlgorithmConfig):
    """List of configurations for Adversarial Motion Priors (AMP) algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.95.
        lmb (float): scalar of lambda return's computation in GAE. Defaults to 0.95.
        policy_learning_rate (float): learning rate which is set to policy solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.000002.
        policy_momentum (float): learning momentum which is set to policy solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.9.
        policy_weight_decay (float): coefficient for weight decay of policy function parameters. \
            In AMP, weight decay is only applied to non bias parameters. Defaults to 0.0005.
        action_bound_loss_coefficient (float): coefficient of action bound loss. Defaults to 10.0
        epsilon (float): probability ratio clipping range of ppo style policy update. Defaults to 0.2
        v_function_learning_rate (float): learning rate which is set to policy solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0005.
        v_function_momentum (float): learning momentum which is set to value function solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.9.
        normalized_advantage_clip (Tuple[float, float]): clipping value for estimated advantages.\
            This clipping is applied after a normalization. Defaults to (-4.0, 4.0)
        value_at_task_fail (float): value for a task fail state. We overwrite the value of the state by this value \
            when computing the value targets. Defaults to 0.0.
        value_at_task_success (float): value for a task success state. We overwrite the value of the state by this \
            value when computing the value targets. Defaults to 1.0.
        target_value_clip (Tuple[float, float]): clipping value for estimated value targets. Defaults to (0.0, 1.0).
        epochs (int): number of epochs to perform in each training iteration for policy and value function. \
            Defaults to 1.
        actor_num (int): number of parallel actors. Defaults to 16.
        batch_size(int): training batch size for policy and value function. Defaults to 256.
        actor_timesteps (int): number of timesteps to interact with the environment by the actors. Defaults to 4096.
        max_explore_steps (int): number of maximum environment exploring steps. Defaults to 200000000.
        final_explore_rate (float): final rate of the environment explorer. Defaults to 0.2.
        timelimit_as_terminal (bool): Treat as done if the environment reaches the \
            `timelimit <https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py>`_.\
            Defaults to False.
        preprocess_state (bool): enable preprocessing the states in the collected experiences\
            before feeding as training batch. Defaults to False.
        state_mean_initializer (Optional[Tuple[Union[float, Tuple[float, ...]], ...]]): \
            mean initialize value for the state preprocessor. Defaults to None.
        state_var_initialize value (Optional[Tuple[Union[float, Tuple[float, ...]], ...]]): \
            variance initializer for the state preprocessor. Defaults to None.
        num_processor_samples (int): number of timesteps for updating the state preprocessor. Defaults to 1000000.
        normalize_action (bool): enable preprocessing the actions. Defaults to False.
        action_mean (Optional[Tuple[float, ...]]) mean for the action normalization. Defaults to None.
        action_var (Optional[Tuple[float, ...]]): variance for the action normalization. Defaults to None.
        discriminator_learning_rate (float): learning rate which is set to discriminator solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.00001.
        discriminator_momentum (float): learning momentum which is set to discriminator solver. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.9.
        discriminator_weight_decay (float): coefficient for weight decay of value function parameters. \
            In AMP, weight decay is only applied to non bias parameters. Defaults to 0.0005.
        discriminator_extra_regularization_coefficient (float): coefficient value of extra regularization \
            of discriminator function parameters that are defined in \
            `discriminator_extra_regularization_variable_names`. Defaults to 0.05.
        discriminator_extra_regularization_variable_names (Tuple[str]): \
            variable names for applying extra regularization. Defaults to ("logits/affine/W",).
        discriminator_gradient_penelty_coefficient (float): coefficient value of gradient penalty.\
            See equation (8) in AMP paper. Defaults to 10.0.
        discriminator_gradient_penalty_indexes (Optional[Tuple[int, ...]]): state index number for \
            applying gradient penalty. Defaults to (1,).
        discriminator_batch_size (int): training batch size for discriminator function Defaults to 256.
        discriminator_epochs (int): number of epochs to perform in each training iteration \
            for discriminator function. Defaults to 2.
        discriminator_reward_scale (float): reward scale.\
            This value will multiply the output reward from the discriminator. Defaults to 2.0.
        discriminator_replay_buffer_size (int): replay buffer size for discriminator training. Defaults to 100000.
        use_reward_from_env (bool): enable to use task reward (i.e., reward from the environment). Defaults to False.
        lerp_reward_coefficient (float): coefficient value for lerping the reward from the environment and \
            the reward from the discriminator. Defaults to 0.5
        act_deterministic_in_eval (bool): enable act deterministically at evalution. Defaults to True.
        seed (int): base seed of random number generator used by the actors. Defaults to 1.
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    gamma: float = 0.95
    lmb: float = 0.95

    policy_learning_rate: float = 0.000002
    policy_momentum: float = 0.9
    policy_weight_decay: float = 0.0005
    action_bound_loss_coefficient: float = 10.0
    epsilon: float = 0.2

    v_function_learning_rate: float = 0.0005
    v_function_momentum: float = 0.9
    normalized_advantage_clip: Tuple[float, float] = (-4.0, 4.0)
    value_at_task_fail: float = 0.0
    value_at_task_success: float = 1.0
    target_value_clip: Tuple[float, float] = (0.0, 1.0)

    epochs: int = 1
    actor_num: int = 16
    batch_size: int = 256
    actor_timesteps: int = 4096

    max_explore_steps: int = 200000000
    final_explore_rate: float = 0.2
    timelimit_as_terminal: bool = False

    preprocess_state: bool = False
    state_mean_initializer: Optional[Tuple[Union[float, Tuple[float, ...]], ...]] = None
    state_var_initializer: Optional[Tuple[Union[float, Tuple[float, ...]], ...]] = None
    num_processor_samples: int = 1000000
    normalize_action: bool = False
    action_mean: Optional[Tuple[float, ...]] = None
    action_var: Optional[Tuple[float, ...]] = None

    discriminator_learning_rate: float = 0.00001
    discriminator_momentum: float = 0.9
    discriminator_weight_decay: float = 0.0005
    discriminator_extra_regularization_coefficient: float = 0.05
    discriminator_extra_regularization_variable_names: Tuple[str] = ("logits/affine/W",)
    discriminator_gradient_penelty_coefficient: float = 10.0
    discriminator_gradient_penalty_indexes: Optional[Tuple[int, ...]] = (1,)
    discriminator_batch_size: int = 256
    discriminator_epochs: int = 2
    discriminator_reward_scale: float = 2.0
    discriminator_agent_replay_buffer_size: int = 100000

    use_reward_from_env: bool = False
    lerp_reward_coefficient: float = 0.5

    act_deterministic_in_eval: bool = True
    seed: int = 1

    def __post_init__(self):
        """__post_init__

        Check the values are in valid range.
        """
        self._assert_between(self.gamma, 0.0, 1.0, "gamma")
        self._assert_between(self.lmb, 0.0, 1.0, "lmb")

        self._assert_positive(self.policy_learning_rate, "policy_learning_rate")
        self._assert_positive(self.policy_momentum, "policy_momentum")
        self._assert_positive(self.policy_weight_decay, "policy_weight_decay")
        self._assert_positive(self.action_bound_loss_coefficient, "action_bound_loss_coefficient")
        self._assert_positive(self.epsilon, "epsilon")

        self._assert_positive(self.v_function_learning_rate, "v_function_learning_rate")
        self._assert_positive(self.v_function_momentum, "v_function_momentum")
        if self.normalized_advantage_clip[0] > self.normalized_advantage_clip[1]:
            raise ValueError("min normalized_advantage_clip is larger than normalized_advantage_clip")

        if self.target_value_clip[0] > self.target_value_clip[1]:
            raise ValueError("min target_value_clip is larger than max target_value_clip")

        self._assert_positive(self.epochs, "epochs")
        self._assert_positive(self.actor_num, "actor num")
        self._assert_positive(self.batch_size, "batch_size")
        self._assert_positive(self.actor_timesteps, "actor_timesteps")

        self._assert_positive(self.max_explore_steps, "max_explore_steps")
        self._assert_between(self.final_explore_rate, 0.0, 1.0, "final_explore_rate")
        self._assert_positive(self.num_processor_samples, "num_processor_samples")

        self._assert_positive(self.discriminator_learning_rate, "discriminator_learning_rate")
        self._assert_positive(self.discriminator_momentum, "discriminator_momentum")
        self._assert_positive(self.discriminator_weight_decay, "discriminator_weight_decay")
        self._assert_positive(
            self.discriminator_extra_regularization_coefficient, "discriminator_extra_regularization_coefficient"
        )
        self._assert_positive(
            self.discriminator_gradient_penelty_coefficient, "discriminator_gradient_penelty_coefficient"
        )
        self._assert_positive(self.discriminator_batch_size, "discriminator_batch_size")
        self._assert_positive(self.discriminator_epochs, "discriminator_epochs")
        self._assert_positive(self.discriminator_reward_scale, "discriminator_reward_scale")
        self._assert_positive(self.discriminator_agent_replay_buffer_size, "discriminator_agent_replay_buffer_size")
        self._assert_between(self.lerp_reward_coefficient, 0.0, 1.0, "lerp_reward_coefficient")


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: AMPConfig,
        **kwargs,
    ) -> StochasticPolicy:
        if env_info.is_goal_conditioned_env():
            return AMPGatedPolicy(scope_name, env_info.action_dim, 0.01)
        else:
            return AMPPolicy(scope_name, env_info.action_dim, 0.01)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: AMPConfig,
        **kwargs,
    ) -> VFunction:
        if env_info.is_goal_conditioned_env():
            return AMPGatedVFunction(scope_name)
        else:
            return AMPVFunction(scope_name)


class DefaultRewardFunctionBuilder(ModelBuilder[RewardFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: AMPConfig,
        **kwargs,
    ) -> RewardFunction:
        return AMPDiscriminator(scope_name, 1.0)


class DefaultVFunctionSolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: AMPConfig, **kwargs
    ) -> nn.solver.Solver:
        return NS.Momentum(lr=algorithm_config.v_function_learning_rate, momentum=algorithm_config.v_function_momentum)


class DefaultPolicySolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: AMPConfig, **kwargs
    ) -> nn.solver.Solver:
        return NS.Momentum(lr=algorithm_config.policy_learning_rate, momentum=algorithm_config.policy_momentum)


class DefaultRewardFunctionSolverBuilder(SolverBuilder):
    def build_solver(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: AMPConfig, **kwargs
    ) -> nn.solver.Solver:
        return NS.Momentum(
            lr=algorithm_config.discriminator_learning_rate, momentum=algorithm_config.discriminator_momentum
        )


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(  # type: ignore[override]
        self,
        env_info: EnvironmentInfo,
        algorithm_config: AMPConfig,
        algorithm: "AMP",
        **kwargs,
    ) -> EnvironmentExplorer:
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            initial_step_num=0,
            timelimit_as_terminal=algorithm_config.timelimit_as_terminal,
            initial_epsilon=1.0,
            final_epsilon=algorithm_config.final_explore_rate,
            max_explore_steps=algorithm_config.max_explore_steps,
            append_explorer_info=True,
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=kwargs["greedy_action_selector"],
            random_action_selector=kwargs["random_action_selector"],
            env_info=env_info,
            config=explorer_config,
        )
        return explorer


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: AMPConfig, **kwargs
    ) -> ReplayBuffer:
        return ReplayBuffer(
            capacity=int(np.ceil(algorithm_config.discriminator_agent_replay_buffer_size / algorithm_config.actor_num))
        )


class AMP(Algorithm):
    """Adversarial Motion Prior (AMP) implementation.

    This class implements the Adversarial Motion Prior (AMP) algorithm
    proposed by Xue Bin Peng, et al. in the paper:
    "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
    For detail see: https://arxiv.org/abs/2104.02180

    This algorithm only supports online training.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`): \
            the environment to train or environment info.
        config (:py:class:`AMPConfig <nnabla_rl.algorithms.ppo.PPOConfig>`): configuration of AMP algorithm.
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`): \
            builder of v function models.
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for v function solvers
        policy_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`): \
            builder of policy models.
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for policy solvers
        reward_function_builder (:py:class:`ModelBuilder[RewardFunction] <nnabla_rl.builders.ModelBuilder>`): \
            builder of reward function models.
        reward_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): \
            builder for reward function solvers.
        state_preprocessor_builder (None or :py:class:`PreprocessorBuilder <nnabla_rl.builders.PreprocessorBuilder>`): \
            state preprocessor builder to preprocess the states.
        env_explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`): \
            builder of environment explorer.
        discriminator_replay_buffer_builder \
        (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`): builder of replay_buffer of \
            discriminator.
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: AMPConfig

    _v_function: VFunction
    _v_function_solver: nn.solver.Solver
    _policy: StochasticPolicy
    _policy_solver: nn.solver.Solver
    _discriminator: RewardFunction
    _discriminator_solver: nn.solver.Solver
    _state_preprocessor: Optional[Preprocessor]
    _discriminator_state_preprocessor: Optional[Preprocessor]

    _policy_trainer: ModelTrainer
    _v_function_trainer: ModelTrainer
    _discriminator_trainer: ModelTrainer

    _policy_solver_builder: SolverBuilder
    _v_solver_builder: SolverBuilder

    _actors: List["_AMPActor"]
    _actor_processes: List[Union[mp.Process, th.Thread]]

    _v_function_trainer_state: Dict[str, Any]
    _policy_trainer_state: Dict[str, Any]
    _discriminator_trainer_state: Dict[str, Any]

    _evaluation_actor: _StochasticPolicyActionSelector

    def __init__(
        self,
        env_or_env_info: Union[gym.Env, EnvironmentInfo],
        config: AMPConfig = AMPConfig(),
        v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
        v_solver_builder: SolverBuilder = DefaultVFunctionSolverBuilder(),
        policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
        policy_solver_builder: SolverBuilder = DefaultPolicySolverBuilder(),
        reward_function_builder: ModelBuilder[RewardFunction] = DefaultRewardFunctionBuilder(),
        reward_solver_builder: SolverBuilder = DefaultRewardFunctionSolverBuilder(),
        state_preprocessor_builder: Optional[PreprocessorBuilder] = None,
        env_explorer_builder: ExplorerBuilder = DefaultExplorerBuilder(),
        discriminator_replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
    ):
        super(AMP, self).__init__(env_or_env_info, config=config)

        # Initialize on cpu and change the context later
        with nn.context_scope(context.get_nnabla_context(-1)):
            policy = policy_builder("pi", self._env_info, self._config)
            v_function = v_function_builder("v", self._env_info, self._config)
            discriminator = reward_function_builder("discriminator", self._env_info, self._config)

            if self._config.preprocess_state:
                if state_preprocessor_builder is None:
                    raise ValueError("State preprocessing is enabled but no preprocessor builder is given")

                self._pi_v_state_preprocessor = state_preprocessor_builder(
                    "pi_v_preprocessor", self._env_info, self._config
                )
                v_function = _StatePreprocessedVFunction(
                    v_function=v_function, preprocessor=self._pi_v_state_preprocessor
                )
                policy = _StatePreprocessedStochasticPolicy(policy=policy, preprocessor=self._pi_v_state_preprocessor)

                self._discriminator_state_preprocessor = state_preprocessor_builder(
                    "r_preprocessor", self._env_info, self._config
                )
                discriminator = _StatePreprocessedRewardFunction(
                    reward_function=discriminator, preprocessor=self._discriminator_state_preprocessor
                )

            self._v_function = v_function
            self._policy = policy
            self._discriminator = discriminator

            self._v_function_solver = v_solver_builder(self._env_info, self._config)
            self._v_solver_builder = v_solver_builder  # keep for later use
            self._policy_solver = policy_solver_builder(self._env_info, self._config)
            self._policy_solver_builder = policy_solver_builder  # keep for later use
            self._discriminator_solver = reward_solver_builder(self._env_info, self._config)
            self._discriminator_solver_builder = reward_solver_builder  # keep for later use
            self._env_explorer_builder = env_explorer_builder  # keep for later use

        self._evaluation_actor = _StochasticPolicyActionSelector(
            self._env_info, self._policy.shallowcopy(), deterministic=self._config.act_deterministic_in_eval
        )
        self._discriminator_agent_replay_buffers = [
            discriminator_replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)
            for _ in range(self._config.actor_num)
        ]
        self._discriminator_expert_replay_buffers = [
            discriminator_replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)
            for _ in range(self._config.actor_num)
        ]

        if self._config.normalize_action:
            action_mean = add_batch_dimension(np.array(self._config.action_mean, dtype=np.float32))
            self._action_mean = nn.Variable.from_numpy_array(action_mean)
            action_var = add_batch_dimension(np.array(self._config.action_var, dtype=np.float32))
            self._action_std = compute_std(
                nn.Variable.from_numpy_array(action_var), epsilon=0.0, mode_for_floating_point_error="max"
            )
        else:
            self._action_mean = None
            self._action_std = None

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._evaluation_action_selector(state, begin_of_episode=begin_of_episode)

            if self._config.normalize_action:
                std = compute_std_ndarray(
                    np.array(self._config.action_var, dtype=np.float32),
                    epsilon=0.0,
                    mode_for_floating_point_error="max",
                )
                action = unnormalize_ndarray(action, np.array(self._config.action_mean, dtype=np.float32), std)

            return action

    def _before_training_start(self, env_or_buffer):
        if not self._is_env(env_or_buffer):
            raise ValueError("AMP only supports online training")

        env = env_or_buffer

        if not (isinstance(env.unwrapped, AMPEnv) or isinstance(env.unwrapped, AMPGoalEnv)):
            raise ValueError("AMP only support AMPEnv and AMPGoalEnv")

        # FIXME: This setup is a workaround for creating underlying model parameters
        # If the parameter is not created, the multiprocessable array (created in launch_actor_processes)
        # will be empty and the agent does not learn anything
        context.set_nnabla_context(-1)
        self._setup_policy_training(env)
        self._setup_v_function_training(env)
        self._setup_reward_function_training(env)

        self._actors, self._actor_processes = self._launch_actor_processes(env)

        context.set_nnabla_context(self._config.gpu_id)

        # Setup again here to use gpu (if it is set)
        old_policy_solver = self._policy_solver
        self._policy_solver = self._policy_solver_builder(self._env_info, self._config)
        self._policy_trainer = self._setup_policy_training(env)
        self._policy_solver.set_states(old_policy_solver.get_states())

        old_v_function_solver = self._v_function_solver
        self._v_function_solver = self._v_solver_builder(self._env_info, self._config)
        self._v_function_trainer = self._setup_v_function_training(env)
        self._v_function_solver.set_states(old_v_function_solver.get_states())

        old_discriminator_solver = self._discriminator_solver
        self._discriminator_solver = self._discriminator_solver_builder(self._env_info, self._config)
        self._discriminator_trainer = self._setup_reward_function_training(env)
        self._discriminator_solver.set_states(old_discriminator_solver.get_states())

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.AMPPolicyTrainerConfig(
            epsilon=self._config.epsilon,
            normalize_action=self._config.normalize_action,
            action_bound_loss_coefficient=self._config.action_bound_loss_coefficient,
            action_mean=self._config.action_mean,
            action_var=self._config.action_var,
            regularization_coefficient=self._config.policy_weight_decay,
        )
        policy_trainer = MT.policy_trainers.AMPPolicyTrainer(
            models=self._policy,
            solvers={self._policy.scope_name: self._policy_solver},
            env_info=self._env_info,
            config=policy_trainer_config,
        )
        return policy_trainer

    def _setup_v_function_training(self, env_or_buffer):
        # training input/loss variables
        v_function_trainer_config = MT.v_value_trainers.MonteCarloVTrainerConfig(
            reduction_method="mean", v_loss_scalar=0.5
        )
        v_function_trainer = MT.v_value_trainers.MonteCarloVTrainer(
            train_functions=self._v_function,
            solvers={self._v_function.scope_name: self._v_function_solver},
            env_info=self._env_info,
            config=v_function_trainer_config,
        )
        return v_function_trainer

    def _setup_reward_function_training(self, env_or_buffer):
        reward_function_trainer_config = MT.reward_trainiers.AMPRewardFunctionTrainerConfig(
            batch_size=self._config.discriminator_batch_size,
            regularization_coefficient=self._config.discriminator_weight_decay,
            extra_regularization_coefficient=self._config.discriminator_extra_regularization_coefficient,
            extra_regularization_variable_names=self._config.discriminator_extra_regularization_variable_names,
            gradient_penelty_coefficient=self._config.discriminator_gradient_penelty_coefficient,
            gradient_penalty_indexes=self._config.discriminator_gradient_penalty_indexes,
        )
        model = (
            self._discriminator._reward_function
            if isinstance(self._discriminator, _StatePreprocessedRewardFunction)
            else self._discriminator
        )
        preprocessor = self._discriminator_state_preprocessor if self._config.preprocess_state else None
        reward_function_trainer = MT.reward_trainiers.AMPRewardFunctionTrainer(
            models=model,
            solvers={self._discriminator.scope_name: self._discriminator_solver},
            env_info=self._env_info,
            state_preprocessor=preprocessor,
            config=reward_function_trainer_config,
        )
        return reward_function_trainer

    def _after_training_finish(self, env_or_buffer):
        for actor in self._actors:
            actor.dispose()
        for process in self._actor_processes:
            self._kill_actor_processes(process)

    def _launch_actor_processes(self, env):
        actors = self._build_amp_actors(
            env=env,
            policy=self._policy,
            v_function=self._v_function,
            state_preprocessor=(self._pi_v_state_preprocessor if self._config.preprocess_state else None),
            reward_function=self._discriminator,
            reward_state_preprocessor=(
                self._discriminator_state_preprocessor if self._config.preprocess_state else None
            ),
            env_explorer=self._env_explorer_builder(
                self._env_info,
                self._config,
                self,
                greedy_action_selector=self._compute_greedy_action,
                random_action_selector=self._compute_explore_action,
            ),
        )
        processes = []
        for actor in actors:
            if self._config.actor_num == 1:
                # Run on same process when we have only 1 actor
                p = th.Thread(target=actor, daemon=False)
            else:
                p = mp.Process(target=actor, daemon=True)
            p.start()
            processes.append(p)
        return actors, processes

    def _build_amp_actors(
        self, env, policy, v_function, state_preprocessor, reward_function, reward_state_preprocessor, env_explorer
    ):
        actors = []
        for i in range(self._config.actor_num):
            actor = _AMPActor(
                actor_num=i,
                env=env,
                env_info=self._env_info,
                policy=policy,
                v_function=v_function,
                state_preprocessor=state_preprocessor,
                reward_function=reward_function,
                reward_state_preprocessor=reward_state_preprocessor,
                config=self._config,
                env_explorer=env_explorer,
            )
            actors.append(actor)
        return actors

    def _run_online_training_iteration(self, env):
        update_interval = self._config.actor_timesteps * self._config.actor_num
        if self.iteration_num % update_interval != 0:
            return

        experiences_per_agent = self._collect_experiences(self._actors)
        assert len(experiences_per_agent) == self._config.actor_num

        self._add_experience_to_reward_buffers(experiences_per_agent)

        # s and s_expert shape is tuple_size * (batch_size, dim)
        s, s_expert = _concatenate_state(experiences_per_agent)

        if update_interval < self.iteration_num:
            # NOTE: The first update (when update_interval == self.iteration_num) will be skipped
            policy_buffers, v_function_buffers = self._create_policy_and_v_function_buffers(experiences_per_agent)
            self._amp_training(
                self._discriminator_agent_replay_buffers,
                self._discriminator_expert_replay_buffers,
                policy_buffers,
                v_function_buffers,
            )

        if self._config.preprocess_state and self.iteration_num < self._config.num_processor_samples:
            self._pi_v_state_preprocessor.update(s)
            self._discriminator_state_preprocessor.update(s)
            self._discriminator_state_preprocessor.update(s_expert)

    def _collect_experiences(self, actors: List["_AMPActor"]):
        def split_result(tuple_val):
            return [_tuple_val for _tuple_val in zip(*tuple_val)]

        for actor in self._actors:
            if self._config.actor_num != 1:
                actor.update_policy_params(self._policy.get_parameters())
                actor.update_reward_function_params(self._discriminator.get_parameters())
                actor.update_v_function_params(self._v_function.get_parameters())
                if self._config.preprocess_state:
                    casted_pi_v_state_preprocessor = cast(Model, self._pi_v_state_preprocessor)
                    actor.update_state_preprocessor_params(casted_pi_v_state_preprocessor.get_parameters())
                    casted_discriminator_state_preprocessor = cast(Model, self._discriminator_state_preprocessor)
                    actor.update_reward_state_preprocessor_params(
                        casted_discriminator_state_preprocessor.get_parameters()
                    )
            else:
                # Its running on same process. No need to synchronize parameters with multiprocessing arrays.
                pass
            actor.run_data_collection()

        experiences_per_agent = []
        for actor in actors:
            result = actor.wait_data_collection()
            # Copy result to main processor
            result = copy.deepcopy(result)

            splitted_result = []
            for r in result:
                if isinstance(r, tuple):
                    splitted_result.append(split_result(r))
                else:
                    splitted_result.append(r)

                assert len(splitted_result[-1]) == self._config.actor_timesteps

            experience = [
                (s, a, r, non_terminal, n_s, log_prob, non_greedy, e_s, e_a, e_s_next, v_target, advantage)
                for (s, a, r, non_terminal, n_s, log_prob, non_greedy, e_s, e_a, e_s_next, v_target, advantage) in zip(
                    *splitted_result
                )
            ]
            assert len(experience) == self._config.actor_timesteps
            experiences_per_agent.append(experience)

        assert len(experiences_per_agent) == self._config.actor_num
        return experiences_per_agent

    def _add_experience_to_reward_buffers(self, experience_per_agent):
        assert len(self._discriminator_agent_replay_buffers) == len(experience_per_agent)
        assert len(self._discriminator_expert_replay_buffers) == len(experience_per_agent)
        for agent_buffer, expert_buffer, experience in zip(
            self._discriminator_agent_replay_buffers, self._discriminator_expert_replay_buffers, experience_per_agent
        ):
            agent_buffer.append_all(experience)
            expert_buffer.append_all(experience)

    def _create_policy_and_v_function_buffers(self, experiences_per_agent):
        policy_buffers = []
        v_function_buffers = []

        for experience in experiences_per_agent:
            policy_buffer = ReplayBuffer()
            v_function_buffer = ReplayBuffer()

            for s, a, r, non_terminal, n_s, log_prob, non_greedy, _, _, _, v, ad in experience:
                v_function_buffer.append((s, a, r, non_terminal, n_s, log_prob, v, ad, non_greedy))
                if non_greedy:  # NOTE: Only use sampled action for policy learning
                    policy_buffer.append((s, a, r, non_terminal, n_s, log_prob, v, ad, non_greedy))

            policy_buffers.append(policy_buffer)
            v_function_buffers.append(v_function_buffer)

        return policy_buffers, v_function_buffers

    def _kill_actor_processes(self, process):
        if isinstance(process, mp.Process):
            process.terminate()
        else:
            # This is a thread. do nothing
            pass
        process.join()

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _amp_training(
        self,
        discriminator_agent_replay_buffers,
        discriminator_expert_replay_buffers,
        policy_replay_buffers,
        v_function_replay_buffers,
    ):
        self._reward_function_training(discriminator_agent_replay_buffers, discriminator_expert_replay_buffers)

        total_updates = (self._config.actor_num * self._config.actor_timesteps) // self._config.batch_size
        self._v_function_training(total_updates, v_function_replay_buffers)
        self._policy_training(total_updates, policy_replay_buffers)

    def _reward_function_training(self, agent_buffers: List[ReplayBuffer], expert_buffers: List[ReplayBuffer]):
        num_updates = (self._config.actor_num * self._config.actor_timesteps) // self._config.discriminator_batch_size
        for _ in range(self._config.discriminator_epochs):
            for _ in range(num_updates):
                agent_experiences = _sample_experiences_from_buffers(
                    agent_buffers, self._config.discriminator_batch_size
                )
                (s_agent, a_agent, _, _, s_next_agent, *_) = marshal_experiences(agent_experiences)

                expert_experiences = _sample_experiences_from_buffers(
                    expert_buffers, self._config.discriminator_batch_size
                )
                (_, _, _, _, _, _, _, s_expert, a_expert, s_next_expert, _, _) = marshal_experiences(expert_experiences)

                extra = {}
                extra["s_current_agent"] = s_agent
                extra["a_current_agent"] = a_agent
                extra["s_next_agent"] = s_next_agent
                extra["s_current_expert"] = s_expert
                extra["a_current_expert"] = a_expert
                extra["s_next_expert"] = s_next_expert

                batch = TrainingBatch(batch_size=self._config.discriminator_batch_size, extra=extra)
                self._discriminator_trainer_state = self._discriminator_trainer.train(batch)

    def _v_function_training(self, total_updates, v_function_replay_buffers):
        v_function_buffer_iterator = _EquallySampleBufferIterator(
            total_updates, v_function_replay_buffers, self._config.batch_size
        )
        for _ in range(self._config.epochs):
            for experiences in v_function_buffer_iterator:
                (s, a, _, _, _, _, v_target, _, _) = marshal_experiences(experiences)
                extra = {}
                extra["v_target"] = v_target
                batch = TrainingBatch(batch_size=len(experiences), s_current=s, a_current=a, extra=extra)
                self._v_function_trainer_state = self._v_function_trainer.train(batch)

    def _policy_training(self, total_updates, policy_replay_buffers):
        policy_buffer_iterator = _EquallySampleBufferIterator(
            total_updates, policy_replay_buffers, self._config.batch_size
        )
        for _ in range(self._config.epochs):
            for experiences in policy_buffer_iterator:
                (s, a, _, _, _, log_prob, _, advantage, _) = marshal_experiences(experiences)
                extra = {}
                extra["log_prob"] = log_prob
                extra["advantage"] = advantage
                batch = TrainingBatch(batch_size=len(experiences), s_current=s, a_current=a, extra=extra)
                self._policy_trainer_state = self._policy_trainer.train(batch)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._v_function.scope_name] = self._v_function
        models[self._discriminator.scope_name] = self._discriminator
        if self._config.preprocess_state and isinstance(self._discriminator_state_preprocessor, Model):
            models[self._discriminator_state_preprocessor.scope_name] = self._discriminator_state_preprocessor
        if self._config.preprocess_state and isinstance(self._pi_v_state_preprocessor, Model):
            models[self._pi_v_state_preprocessor.scope_name] = self._pi_v_state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._v_function.scope_name] = self._v_function_solver
        solvers[self._policy.scope_name] = self._policy_solver
        solvers[self._discriminator.scope_name] = self._discriminator_solver
        return solvers

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = (
            EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) else env_or_env_info
        )
        return not env_info.is_discrete_action_env() and not env_info.is_tuple_action_env()

    @eval_api
    def _compute_greedy_action(self, s, *, begin_of_episode=False):
        s = add_batch_dimension(s)
        if not hasattr(self, "_greedy_state_var"):
            self._greedy_state_var = create_variable(1, self._env_info.state_shape)
            distribution = self._policy.pi(self._greedy_state_var)
            self._greedy_action = distribution.choose_probable()
            self._greedy_action_log_prob = distribution.log_prob(self._greedy_action)
            if self._config.normalize_action:
                # NOTE: an action from policy is normalized.
                self._greedy_action = unnormalize(self._greedy_action, self._action_mean, self._action_std)

        set_data_to_variable(self._greedy_state_var, s)
        nn.forward_all([self._greedy_action, self._greedy_action_log_prob])
        action = np.squeeze(self._greedy_action.d, axis=0)
        log_prob = np.squeeze(self._greedy_action_log_prob.d, axis=0)
        info = {}
        info["log_prob"] = log_prob
        return action, info

    @eval_api
    def _compute_explore_action(self, s, *, begin_of_episode=False):
        s = add_batch_dimension(s)
        if not hasattr(self, "_explore_state_var"):
            self._explore_state_var = create_variable(1, self._env_info.state_shape)
            distribution = self._policy.pi(self._explore_state_var)
            self._explore_action, self._explore_action_log_prob = distribution.sample_and_compute_log_prob()
            if self._config.normalize_action:
                # NOTE: an action from policy is normalized.
                self._explore_action = unnormalize(self._explore_action, self._action_mean, self._action_std)

        set_data_to_variable(self._explore_state_var, s)
        nn.forward_all([self._explore_action, self._explore_action_log_prob])
        action = np.squeeze(self._explore_action.d, axis=0)
        log_prob = np.squeeze(self._explore_action_log_prob.d, axis=0)
        info = {}
        info["log_prob"] = log_prob
        return action, info

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(AMP, self).latest_iteration_state

        if hasattr(self, "_discriminator_trainer_state"):
            discriminator_trainer_state = {}
            for k, v in self._discriminator_trainer_state.items():
                discriminator_trainer_state[k] = float(v)
            latest_iteration_state["scalar"].update(discriminator_trainer_state)

        if hasattr(self, "_v_function_trainer_state"):
            latest_iteration_state["scalar"].update({"v_loss": float(self._v_function_trainer_state["v_loss"])})

        if hasattr(self, "_policy_trainer_state"):
            policy_trainer_state = {}
            for k, v in self._policy_trainer_state.items():
                policy_trainer_state[k] = float(v)
            latest_iteration_state["scalar"].update(policy_trainer_state)

        return latest_iteration_state

    @property
    def trainers(self):
        return {
            "discriminator": self._discriminator_trainer,
            "v_function": self._v_function_trainer,
            "policy": self._policy_trainer,
        }


def _sample_experiences_from_buffers(buffers: List[ReplayBuffer], batch_size: int) -> List[Experience]:
    experiences: List[Experience] = []
    for buffer in buffers:
        experience, _ = buffer.sample(num_samples=int(np.ceil(batch_size / len(buffers))))
        experience = cast(List[Experience], experience)
        experiences.extend(experience)

    assert len(experiences) >= batch_size
    return experiences[:batch_size]


def _concatenate_state(experiences_per_agent) -> Tuple[np.ndarray, np.ndarray]:
    all_experience = []
    for e in experiences_per_agent:
        all_experience.extend(e)
    s, _, _, _, _, _, _, e_s, _, _, _, _ = marshal_experiences(all_experience)
    return s, e_s


class _AMPActor:

    def __init__(
        self,
        actor_num: int,
        env: gym.Env,
        env_info: EnvironmentInfo,
        policy: StochasticPolicy,
        v_function: VFunction,
        state_preprocessor: Optional[Preprocessor],
        reward_function: RewardFunction,
        reward_state_preprocessor: Optional[Preprocessor],
        env_explorer: EnvironmentExplorer,
        config: AMPConfig,
    ):
        # These variables will be copied when process is created
        self._actor_num = actor_num
        self._env = env
        self._env_info = env_info
        self._policy = policy
        self._v_function = v_function
        self._reward_function = reward_function
        self._reward_state_preprocessor = reward_state_preprocessor
        self._state_preprocessor = state_preprocessor
        self._timesteps = config.actor_timesteps
        self._gamma = config.gamma
        self._lambda = config.lmb
        self._config = config
        self._env_explorer = env_explorer

        # IPC communication variables
        self._disposed = mp.Value("i", False)
        self._task_start_event = mp.Event()
        self._task_finish_event = mp.Event()

        self._policy_mp_arrays = new_mp_arrays_from_params(policy.get_parameters())
        self._v_function_mp_arrays = new_mp_arrays_from_params(v_function.get_parameters())
        self._reward_function_mp_arrays = new_mp_arrays_from_params(reward_function.get_parameters())
        if self._config.preprocess_state:
            assert state_preprocessor is not None
            casted_state_preprocessor = cast(Model, state_preprocessor)
            self._state_preprocessor_mp_arrays = new_mp_arrays_from_params(casted_state_preprocessor.get_parameters())

            assert reward_state_preprocessor is not None
            casted_reward_state_preprocessor = cast(Model, reward_state_preprocessor)
            self._reward_state_preprocessor_mp_arrays = new_mp_arrays_from_params(
                casted_reward_state_preprocessor.get_parameters()
            )

        MultiProcessingArrays = namedtuple(
            "MultiProcessingArrays",
            [
                "state",
                "action",
                "reward",
                "non_terminal",
                "next_state",
                "log_prob",
                "non_greedy_action",
                "expert_state",
                "expert_action",
                "expert_next_state",
                "v_target",
                "advantage",
            ],
        )

        state_mp_array = self._prepare_state_mp_array(env_info.observation_space, env_info)
        action_mp_array = self._prepare_action_mp_array(env_info.action_space, env_info)

        scalar_mp_array_shape = (self._timesteps, 1)
        reward_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )
        non_terminal_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )
        next_state_mp_array = self._prepare_state_mp_array(env_info.observation_space, env_info)
        log_prob_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )
        non_greedy_action_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )
        v_target_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )
        advantage_mp_array = (
            mp_array_from_np_array(np.empty(shape=scalar_mp_array_shape, dtype=np.float32)),
            scalar_mp_array_shape,
            np.float32,
        )

        expert_state_mp_array = self._prepare_state_mp_array(env_info.observation_space, env_info)
        expert_action_mp_array = self._prepare_action_mp_array(env_info.action_space, env_info)
        expert_next_state_mp_array = self._prepare_state_mp_array(env_info.observation_space, env_info)

        self._mp_arrays = MultiProcessingArrays(
            state_mp_array,
            action_mp_array,
            reward_mp_array,
            non_terminal_mp_array,
            next_state_mp_array,
            log_prob_mp_array,
            non_greedy_action_mp_array,
            expert_state_mp_array,
            expert_action_mp_array,
            expert_next_state_mp_array,
            v_target_mp_array,
            advantage_mp_array,
        )

        self._reward_min = np.inf
        self._reward_max = -np.inf

    def __call__(self):
        self._run_actor_loop()

    def dispose(self):
        self._disposed.value = True
        self._task_start_event.set()

    def run_data_collection(self):
        self._task_finish_event.clear()
        self._task_start_event.set()

    def wait_data_collection(self):
        def _mp_to_np_array(mp_array):
            if isinstance(mp_array[0], tuple):
                # tupled state
                return tuple(mp_to_np_array(*array) for array in mp_array)
            else:
                return mp_to_np_array(*mp_array)

        self._task_finish_event.wait()
        return tuple(_mp_to_np_array(mp_array) for mp_array in self._mp_arrays)

    def update_policy_params(self, params):
        self._update_params(src=params, dest=self._policy_mp_arrays)

    def update_v_function_params(self, params):
        self._update_params(src=params, dest=self._v_function_mp_arrays)

    def update_reward_function_params(self, params):
        self._update_params(src=params, dest=self._reward_function_mp_arrays)

    def update_state_preprocessor_params(self, params):
        self._update_params(src=params, dest=self._state_preprocessor_mp_arrays)

    def update_reward_state_preprocessor_params(self, params):
        self._update_params(src=params, dest=self._reward_state_preprocessor_mp_arrays)

    def _update_params(self, src, dest):
        copy_params_to_mp_arrays(src, dest)

    def _run_actor_loop(self):
        context.set_nnabla_context(self._config.gpu_id)
        if self._config.seed >= 0:
            seed = self._actor_num + self._config.seed
        else:
            seed = os.getpid()

        self._env.seed(seed)
        set_global_seed(seed)
        while True:
            self._task_start_event.wait()
            if self._disposed.get_obj():
                break
            if self._config.actor_num != 1:
                # Running on different process
                # Sync parameters through multiproccess arrays
                self._synchronize_policy_params(self._policy.get_parameters())
                self._synchronize_v_function_params(self._v_function.get_parameters())
                self._synchronize_reward_function_params(self._reward_function.get_parameters())

                if self._config.preprocess_state:
                    self._synchronize_preprocessor_params(self._state_preprocessor.get_parameters())
                    self._synchronize_reward_preprocessor_params(self._reward_state_preprocessor.get_parameters())

            experiences = self._run_data_collection()
            self._fill_result(experiences)

            self._task_start_event.clear()
            self._task_finish_event.set()

    def _fill_result(self, experiences):
        indexes = np.arange(len(experiences))
        drng.shuffle(indexes)
        experiences = [experiences[i] for i in indexes[: self._config.actor_timesteps]]
        (s, a, r, non_terminal, s_next, log_prob, non_greedy_action, e_s, e_a, e_s_next, v_target, advantage) = (
            marshal_experiences(experiences)
        )

        _copy_np_array_to_mp_array(s, self._mp_arrays.state)
        _copy_np_array_to_mp_array(a, self._mp_arrays.action)
        _copy_np_array_to_mp_array(r, self._mp_arrays.reward)
        _copy_np_array_to_mp_array(non_terminal, self._mp_arrays.non_terminal)
        _copy_np_array_to_mp_array(s_next, self._mp_arrays.next_state)
        _copy_np_array_to_mp_array(log_prob, self._mp_arrays.log_prob)
        _copy_np_array_to_mp_array(non_greedy_action, self._mp_arrays.non_greedy_action)
        _copy_np_array_to_mp_array(e_s, self._mp_arrays.expert_state)
        _copy_np_array_to_mp_array(e_a, self._mp_arrays.expert_action)
        _copy_np_array_to_mp_array(e_s_next, self._mp_arrays.expert_next_state)
        _copy_np_array_to_mp_array(v_target, self._mp_arrays.v_target)
        _copy_np_array_to_mp_array(advantage, self._mp_arrays.advantage)

    def _run_data_collection(self):
        experiences = self._env_explorer.step(self._env, n=self._timesteps)
        rewards = self._compute_rewards(experiences)
        self._reward_min = min(np.min(rewards), self._reward_min)
        self._reward_max = max(np.max(rewards), self._reward_max)
        v_targets, advantages = _compute_v_target_and_advantage_with_clipping_and_overwriting(
            v_function=self._v_function,
            experiences=experiences,
            rewards=rewards,
            gamma=self._config.gamma,
            lmb=self._config.lmb,
            value_clip=(self._reward_min / (1.0 - self._config.gamma), self._reward_max / (1.0 - self._config.gamma)),
            value_at_task_fail=self._config.value_at_task_fail,
            value_at_task_success=self._config.value_at_task_success,
        )
        assert self._config.target_value_clip[0] < self._config.target_value_clip[1]
        v_targets = np.clip(v_targets, a_min=self._config.target_value_clip[0], a_max=self._config.target_value_clip[1])

        advantage_std = compute_std_ndarray(np.var(advantages), epsilon=1e-5, mode_for_floating_point_error="add")
        advantages = normalize_ndarray(
            advantages, mean=np.mean(advantages), std=advantage_std, value_clip=self._config.normalized_advantage_clip
        )

        assert len(experiences) == len(v_targets)
        assert len(experiences) == len(advantages)
        organized_experiences = []
        for (s, a, r, non_terminal, s_next, info), v_target, advantage in zip(experiences, v_targets, advantages):
            expert_s, expert_a, _, _, expert_n_s, _ = info["expert_experience"]

            assert "greedy_action" in info
            organized_experiences.append(
                (
                    s,
                    a,
                    r,
                    non_terminal,
                    s_next,
                    info["log_prob"],
                    0.0 if info["greedy_action"] else 1.0,
                    expert_s,
                    expert_a,
                    expert_n_s,
                    v_target,
                    advantage,
                )
            )

        return organized_experiences

    def _compute_rewards(self, experiences: List[Experience]) -> List[float]:
        if not hasattr(self, "_reward_var"):
            self._s_var_label = create_variable(1, self._env_info.state_shape)
            self._s_next_var_label = create_variable(1, self._env_info.state_shape)
            self._a_var_label = create_variable(1, self._env_info.action_shape)
            logits = self._reward_function.r(self._s_var_label, self._a_var_label, self._s_next_var_label)
            # equation (7) in the paper
            self._reward_var = 1.0 - 0.25 * ((logits - 1.0) ** 2)
            self._reward_var = NF.maximum_scalar(self._reward_var, val=0.0) * self._config.discriminator_reward_scale

        rewards: List[float] = []
        for experience in experiences:
            s, a, env_r, _, n_s, *_ = experience
            set_data_to_variable(self._s_var_label, s)
            set_data_to_variable(self._a_var_label, a)
            set_data_to_variable(self._s_next_var_label, n_s)
            self._reward_var.forward()

            if self._config.use_reward_from_env:
                reward = (1.0 - self._config.lerp_reward_coefficient) * float(
                    self._reward_var.d
                ) + self._config.lerp_reward_coefficient * float(env_r)
            else:
                reward = float(self._reward_var.d)
            rewards.append(reward)

        return rewards

    def _synchronize_v_function_params(self, params):
        self._synchronize_params(src=self._v_function_mp_arrays, dest=params)

    def _synchronize_policy_params(self, params):
        self._synchronize_params(src=self._policy_mp_arrays, dest=params)

    def _synchronize_reward_function_params(self, params):
        self._synchronize_params(src=self._reward_function_mp_arrays, dest=params)

    def _synchronize_preprocessor_params(self, params):
        self._synchronize_params(src=self._state_preprocessor_mp_arrays, dest=params)

    def _synchronize_reward_preprocessor_params(self, params):
        self._synchronize_params(src=self._reward_state_preprocessor_mp_arrays, dest=params)

    def _synchronize_params(self, src, dest):
        copy_mp_arrays_to_params(src, dest)

    def _prepare_state_mp_array(self, obs_space, env_info):
        if env_info.is_tuple_state_env():
            state_mp_arrays = []
            state_mp_array_shapes = []
            state_mp_array_dtypes = []
            for space in obs_space:
                state_mp_array_shape = (self._timesteps, *space.shape)
                state_mp_array = mp_array_from_np_array(np.empty(shape=state_mp_array_shape, dtype=space.dtype))
                state_mp_array_shapes.append(state_mp_array_shape)
                state_mp_array_dtypes.append(space.dtype)
                state_mp_arrays.append(state_mp_array)
            return tuple(x for x in zip(state_mp_arrays, state_mp_array_shapes, state_mp_array_dtypes))
        else:
            state_mp_array_shape = (self._timesteps, *obs_space.shape)
            state_mp_array = mp_array_from_np_array(np.empty(shape=state_mp_array_shape, dtype=obs_space.dtype))
            return (state_mp_array, state_mp_array_shape, obs_space.dtype)

    def _prepare_action_mp_array(self, action_space, env_info):
        action_mp_array_shape = (self._timesteps, action_space.shape[0])
        action_mp_array = mp_array_from_np_array(np.empty(shape=action_mp_array_shape, dtype=action_space.dtype))
        return (action_mp_array, action_mp_array_shape, action_space.dtype)


def _copy_np_array_to_mp_array(
    np_array: Union[np.ndarray, Tuple[np.ndarray]],
    mp_array_shape_type: Union[
        Tuple[np.ndarray, Tuple[int, ...], np.dtype], Tuple[Tuple[np.ndarray, Tuple[int, ...], np.dtype]]
    ],
):
    """Copy numpy array to multiprocessing array.

    Args:
        np_array(Union[np.ndarray, Tuple[np.ndarray]]): copy source of numpy array.
        mp_array_shape_type
            (Union[Tuple[np.ndarray, Tuple[int, ...], np.dtype], Tuple[Tuple[np.ndarray, Tuple[int, ...], np.dtype]]]):
            copy target of multiprocessing array, shape and type.
    """
    if isinstance(np_array, tuple) and isinstance(mp_array_shape_type[0], tuple):
        assert len(np_array) == len(mp_array_shape_type)
        for np_ary, mp_ary_shape_type in zip(np_array, mp_array_shape_type):
            np_to_mp_array(np_ary, mp_ary_shape_type[0], mp_ary_shape_type[2])
    elif isinstance(np_array, np.ndarray) and isinstance(mp_array_shape_type[0], np.ndarray):
        np_to_mp_array(np_array, mp_array_shape_type[0], mp_array_shape_type[2])
    else:
        raise ValueError("Invalid pair of np_array and mp_array!")


def _compute_v_target_and_advantage_with_clipping_and_overwriting(
    v_function: VFunction,
    experiences: List[Experience],
    rewards: List[float],
    gamma: float,
    lmb: float,
    value_at_task_fail: float,
    value_at_task_success: float,
    value_clip: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(v_function, VFunction), "Invalid v_function"
    if value_clip is not None:
        assert value_clip[0] < value_clip[1]
    assert value_at_task_success > value_at_task_fail
    assert len(rewards) == len(experiences)

    T = len(experiences)
    v_targets: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantages: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantage: np.float32 = np.float32(0.0)

    v_current = None
    v_next = None
    s_var = create_variable(1, _get_shape(experiences[0][0]))
    v = v_function.v(s_var)  # build graph
    v_forwards = []

    for t in reversed(range(T)):
        # Not use reward from the environment
        s_current, _, _, non_terminal, s_next, info, *_ = experiences[t]
        r = rewards[t]

        if not non_terminal:
            v_next = None
            advantage = np.float32(0.0)

        # predict current v
        set_data_to_variable(s_var, s_current)
        v.forward()
        v_current = np.squeeze(v.d)
        v_forwards.append(v_current)

        if value_clip is not None:
            v_current = np.clip(v_current, a_min=value_clip[0], a_max=value_clip[1])

        if v_next is None:
            set_data_to_variable(s_var, s_next)
            v.forward()
            v_next = np.squeeze(v.d)

            if value_clip is not None:
                v_next = np.clip(v_next, a_min=value_clip[0], a_max=value_clip[1])

            if info["task_result"] == TaskResult.SUCCESS:
                assert not non_terminal
                v_next = value_at_task_success
            elif info["task_result"] == TaskResult.FAIL:
                assert not non_terminal
                v_next = value_at_task_fail
            elif info["task_result"] == TaskResult.UNKNOWN:
                pass
            else:
                raise ValueError

        delta = r + gamma * v_next - v_current
        advantage = np.float32(delta + gamma * lmb * advantage)
        # A = Q - V, V = E[Q] -> v_target = A + V
        v_target = advantage + v_current

        v_targets[t] = v_target
        advantages[t] = advantage

        v_next = v_current

    return np.array(v_targets, dtype=np.float32), np.array(advantages, dtype=np.float32)


class _EndlessBufferIterator(BufferIterator):
    """This buffer iterates endlessly."""

    def __init__(self, buffer, batch_size, shuffle=True):
        super().__init__(buffer, batch_size, shuffle, repeat=True)

    def next(self):
        indices = self._indices[self._index : self._index + self._batch_size]
        if len(indices) < self._batch_size:
            rest = self._batch_size - len(indices)
            self.reset()
            indices = np.append(indices, self._indices[self._index : self._index + rest])
            self._index += rest
        else:
            self._index += self._batch_size
        return self._sample(indices)

    __next__ = next


class _EquallySampleBufferIterator:
    def __init__(self, total_num_iterations: int, replay_buffers: List[ReplayBuffer], batch_size: int):
        buffer_batch_size = int(np.ceil(batch_size / len(replay_buffers)))
        self._internal_iterators = [
            _EndlessBufferIterator(buffer=buffer, shuffle=False, batch_size=buffer_batch_size)
            for buffer in replay_buffers
        ]
        self._total_num_iterations = total_num_iterations
        self._replay_buffers = replay_buffers
        self._batch_size = batch_size
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        self._num_iterations += 1

        if self._num_iterations > self._total_num_iterations:
            raise StopIteration

        return self._sample()

    __next__ = next

    def reset(self):
        self._num_iterations = 0
        for iterator in self._internal_iterators:
            iterator.reset()

    def _sample(self):
        experiences = []
        for iterator in self._internal_iterators:
            experience, *_ = iterator.next()
            experiences.extend(experience)

        if len(experiences) > self._batch_size:
            drng.shuffle(experiences)
            experiences = experiences[: self._batch_size]

        return experiences
