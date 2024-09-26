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
from typing import Dict, List, Union

import gym

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import _StochasticPolicyActionSelector
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import IQLPolicy, IQLQFunction, IQLVFunction, QFunction, StochasticPolicy, VFunction
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class IQLConfig(AlgorithmConfig):
    """List of configurations for IQL algorithm.

    Args:
        gamma (float): discount factor of reward. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0003.
        batch_size (int): training batch size. Defaults to 256.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        expectile (float): the expectile value for expectile regression loss. Defaults to 0.7.
        beta (float): the temperature parameter of advantage weight. Defaults to 3.0
        advantage_clip (Optional[float]): the value for clipping advantage weight. Defaults to 100.0
    """

    gamma: float = 0.99
    learning_rate: float = 3.0 * 1e-4
    batch_size: int = 256
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    advantage_clip: float = 100.0

    def __post_init__(self):
        """__post_init__

        Check set values are in valid range.
        """
        self._assert_between(self.gamma, 0.0, 1.0, "gamma")
        self._assert_between(self.tau, 0.0, 1.0, "tau")
        self._assert_positive(self.batch_size, "batch_size")
        self._assert_between(self.expectile, 0.0, 1.0, "expectile")


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: IQLConfig,
        **kwargs,
    ) -> QFunction:
        return IQLQFunction(scope_name)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: IQLConfig,
        **kwargs,
    ) -> VFunction:
        return IQLVFunction(scope_name)


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: IQLConfig,
        **kwargs,
    ) -> StochasticPolicy:
        return IQLPolicy(scope_name, action_dim=env_info.action_dim)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self, env_info: EnvironmentInfo, algorithm_config: IQLConfig, **kwargs):  # type: ignore[override]
        return NS.Adam(alpha=algorithm_config.learning_rate)


class IQL(Algorithm):
    """Implicit Q-learning (IQL) algorithm.

    This class implements the Implicit Q-learning (IQL) algorithm
    proposed by I. Kostrikov, et al. in the paper: "OFFLINE REINFORCEMENT LEARNING WITH IMPLICIT Q-LEARNING"
    For details see: https://arxiv.org/abs/2110.06169

    This algorithm only supports offline training.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`IQLConfig <nnabla_rl.algorithms.iql.IQLConfig>`):
            configuration of the IQL algorithm
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q-function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for q-function solvers
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v-function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for v-function solvers
        policy_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for policy solvers
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: IQLConfig
    _train_q_functions: List[QFunction]
    _train_q_solvers: Dict[str, nn.solver.Solver]
    _target_q_functions: List[QFunction]
    _v_function: VFunction
    _v_solver: nn.solver.Solver
    _pi: StochasticPolicy
    _pi_solver: nn.solver.Solver

    def __init__(
        self,
        env_or_env_info: Union[gym.Env, EnvironmentInfo],
        config: IQLConfig = IQLConfig(),
        q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
        q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
        v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
        policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
        policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
    ):
        super(IQL, self).__init__(env_or_env_info, config=config)
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._train_q_functions = self._build_q_functions(q_function_builder)
            self._train_q_solvers = {
                q.scope_name: q_solver_builder(self._env_info, self._config) for q in self._train_q_functions
            }
            self._target_q_functions = [q.deepcopy("target_" + q.scope_name) for q in self._train_q_functions]

            self._v_function = v_function_builder(
                scope_name="v", env_info=self._env_info, algorithm_config=self._config
            )
            self._v_solver = v_solver_builder(self._env_info, self._config)

            self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
            self._pi_solver = policy_solver_builder(self._env_info, self._config)

        self._evaluation_actor = self._setup_evaluation_actor()

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._evaluation_action_selector(state, begin_of_episode=begin_of_episode)
            return action

    def _setup_evaluation_actor(self):
        return _StochasticPolicyActionSelector(self._env_info, self._pi.shallowcopy(), deterministic=True)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _build_q_functions(self, q_function_builder):
        q_functions = []
        for i in range(2):
            q = q_function_builder(scope_name=f"q{i+1}", env_info=self._env_info, algorithm_config=self._config)
            q_functions.append(q)
        return q_functions

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.AWRPolicyTrainerConfig(
            beta=self._config.beta, advantage_clip=self._config.advantage_clip
        )
        policy_trainer = MT.policy_trainers.AWRPolicyTrainer(
            models=self._pi,
            solvers={self._pi.scope_name: self._pi_solver},
            q_functions=self._target_q_functions,
            v_function=self._v_function,
            env_info=self._env_info,
            config=policy_trainer_config,
        )
        return policy_trainer

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_config = MT.q_value_trainers.VTargetedQTrainerConfig(
            reduction_method="mean", q_loss_scalar=1.0
        )
        q_function_trainer = MT.q_value_trainers.VTargetedQTrainer(
            train_functions=self._train_q_functions,
            solvers=self._train_q_solvers,
            target_functions=self._v_function,
            env_info=self._env_info,
            config=q_function_trainer_config,
        )
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q, 1.0)
        return q_function_trainer

    def _setup_v_function_training(self, env_or_buffer):
        v_function_trainer_config = MT.v_value_trainers.IQLVFunctionTrainerConfig(expectile=self._config.expectile)
        v_function_trainer = MT.v_value_trainers.IQLVFunctionTrainer(
            models=self._v_function,
            solvers={self._v_function.scope_name: self._v_solver},
            target_functions=self._target_q_functions,
            env_info=self._env_info,
            config=v_function_trainer_config,
        )
        return v_function_trainer

    def _run_online_training_iteration(self, env):
        # Not support online training
        raise NotImplementedError

    def _run_offline_training_iteration(self, buffer):
        self._iql_training(buffer)

    def _iql_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._config.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshal_experiences(experiences)
        batch = TrainingBatch(
            batch_size=self._config.batch_size,
            s_current=s,
            a_current=a,
            gamma=self._config.gamma,
            reward=r,
            non_terminal=non_terminal,
            s_next=s_next,
            weight=info["weights"],
        )

        self._v_function_trainer_state = self._v_function_trainer.train(batch)

        self._policy_trainer_state = self._policy_trainer.train(batch)

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q, tau=self._config.tau)
        td_errors = self._q_function_trainer_state["td_errors"]
        replay_buffer.update_priorities(td_errors)

    def _models(self):
        models = [self._v_function, *self._train_q_functions, self._pi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers[self._pi.scope_name] = self._pi_solver
        solvers[self._v_function.scope_name] = self._v_solver
        solvers.update(self._train_q_solvers)
        return solvers

    @classmethod
    def is_rnn_supported(self):
        return False

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = (
            EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) else env_or_env_info
        )
        return not env_info.is_discrete_action_env() and not env_info.is_tuple_action_env()

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(IQL, self).latest_iteration_state
        if hasattr(self, "_v_function_trainer_state"):
            latest_iteration_state["scalar"].update({"v_loss": float(self._v_function_trainer_state["v_loss"])})
        if hasattr(self, "_q_function_trainer_state"):
            latest_iteration_state["scalar"].update({"q_loss": float(self._q_function_trainer_state["q_loss"])})
            latest_iteration_state["histogram"].update(
                {"td_errors": self._q_function_trainer_state["td_errors"].flatten()}
            )
        if hasattr(self, "_policy_trainer_state"):
            latest_iteration_state["scalar"].update({"pi_loss": float(self._policy_trainer_state["pi_loss"])})
        return latest_iteration_state

    @property
    def trainers(self):
        return {
            "v_function": self._v_function_trainer,
            "q_function": self._q_function_trainer,
            "policy": self._policy_trainer,
        }
