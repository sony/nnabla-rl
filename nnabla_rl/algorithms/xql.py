# Copyright 2023 Sony Group Corporation.
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
from typing import Any, Dict, List, Union

import gym

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import _StochasticPolicyActionSelector
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import QFunction, StochasticPolicy, VFunction, XQLPolicy, XQLQFunction, XQLVFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class XQLConfig(AlgorithmConfig):
    """XQLConfig List of configurations for XQL algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0003.
        batch_size(int): training batch size. Defaults to 256.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        value_temperature (float): Temperature parameter used to balance between the reward and kl-divergence \
            (a.k.a beta). This parameter will be used in the training of value function. Defaults to 2.0.
            Theoretically, value temperature and policy temperature should be the same. \
            But these two values are introduced for engineering purpose.
        policy_temperature (float): Temperature parameter used to balance between the reward and kl-divergence \
            (a.k.a beta). This parameter will be used in the training of policy function. Defaults to 2.0. \
            Theoretically, value temperature and policy temperature should be the same. \
            But these two values are introduced for engineering purpose.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        num_steps (int): number of steps for N-step Q targets. Defaults to 1.
        pi_unroll_steps (int): Number of steps to unroll policy's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        pi_burn_in_steps (int): Number of burn-in steps to initiaze policy's recurrent layer states during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        pi_reset_rnn_on_terminal (bool): Reset policy's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to True.
        q_unroll_steps (int): Number of steps to unroll q-function's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        q_burn_in_steps (int): Number of burn-in steps to initiaze q-function's recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        q_reset_rnn_on_terminal (bool): Reset q-function's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to True.
        v_unroll_steps (int): Number of steps to unroll v-function's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        v_burn_in_steps (int): Number of burn-in steps to initiaze v-function's recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        v_reset_rnn_on_terminal (bool): Reset v-function's recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to True.
    """

    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    batch_size: int = 256
    tau: float = 0.005
    value_temperature: float = 2.0
    policy_temperature: float = 2.0
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    num_steps: int = 1

    # rnn model support
    pi_unroll_steps: int = 1
    pi_burn_in_steps: int = 0
    pi_reset_rnn_on_terminal: bool = True

    q_unroll_steps: int = 1
    q_burn_in_steps: int = 0
    q_reset_rnn_on_terminal: bool = True

    v_unroll_steps: int = 1
    v_burn_in_steps: int = 0
    v_reset_rnn_on_terminal: bool = True

    def __post_init__(self):
        """__post_init__ Check set values are in valid range."""
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.num_steps, 'num_steps')

        self._assert_positive(self.pi_unroll_steps, 'pi_unroll_steps')
        self._assert_positive_or_zero(self.pi_burn_in_steps, 'pi_burn_in_steps')
        self._assert_positive(self.q_unroll_steps, 'q_unroll_steps')
        self._assert_positive_or_zero(self.q_burn_in_steps, 'q_burn_in_steps')
        self._assert_positive(self.v_unroll_steps, 'v_unroll_steps')
        self._assert_positive_or_zero(self.v_burn_in_steps, 'v_burn_in_steps')


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: XQLConfig,
                    **kwargs) -> QFunction:
        return XQLQFunction(scope_name)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: XQLConfig,
                    **kwargs) -> VFunction:
        return XQLVFunction(scope_name)


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: XQLConfig,
                    **kwargs) -> StochasticPolicy:
        return XQLPolicy(scope_name, env_info.action_dim)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: XQLConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: XQLConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: XQLConfig,
                       algorithm: "XQL",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.RawPolicyExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            initial_step_num=algorithm.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=algorithm._exploration_action_selector,
                                        env_info=env_info,
                                        config=explorer_config)
        return explorer


class XQL(Algorithm):
    """EXtreme Q-Learning (XQL) algorithm implementation.

    This class implements the eXtreme Q-Learning (XQL) algorithm
    proposed by D. Garg, et. al. in the paper: "Extreme Q-Learning: MaxEnt RL without Entropy"
    For detail see: https://arxiv.org/abs/2301.02328

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`XQLConfig <nnabla_rl.algorithms.sac.XQLConfig>`): configuration of the XQL algorithm
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of q function solvers
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of v function solvers
        policy_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of policy solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: XQLConfig
    _train_q_functions: List[QFunction]
    _train_q_solvers: Dict[str, nn.solver.Solver]
    _target_q_functions: List[QFunction]

    _train_v_functions: List[VFunction]
    _train_v_solvers: Dict[str, nn.solver.Solver]

    _pi: StochasticPolicy
    _replay_buffer: ReplayBuffer

    _explorer_builder: ExplorerBuilder
    _environment_explorer: EnvironmentExplorer
    _policy_trainer: ModelTrainer
    _q_function_trainer: ModelTrainer
    _v_function_trainer: ModelTrainer

    _policy_trainer_state: Dict[str, Any]
    _v_function_trainer_state: Dict[str, Any]
    _q_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: XQLConfig = XQLConfig(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(XQL, self).__init__(env_or_env_info, config=config)

        self._explorer_builder = explorer_builder

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._q1 = q_function_builder(scope_name="q1", env_info=self._env_info, algorithm_config=self._config)
            self._q2 = q_function_builder(scope_name="q2", env_info=self._env_info, algorithm_config=self._config)

            self._train_q_functions = [self._q1, self._q2]
            self._train_q_solvers = {}
            for q in self._train_q_functions:
                self._train_q_solvers[q.scope_name] = q_solver_builder(
                    env_info=self._env_info, algorithm_config=self._config)
            self._target_q_functions = [q.deepcopy('target_' + q.scope_name) for q in self._train_q_functions]

            self._v_function = v_function_builder("v", env_info=self._env_info, algorithm_config=self._config)
            self._v_solver = v_solver_builder(self._env_info, self._config)

            self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
            self._pi_solver = policy_solver_builder(self._env_info, self._config)

            self._replay_buffer = replay_buffer_builder(self._env_info, self._config)

        self._evaluation_actor = _StochasticPolicyActionSelector(
            self._env_info, self._pi.shallowcopy(), deterministic=True)
        self._exploration_actor = _StochasticPolicyActionSelector(
            self._env_info, self._pi.shallowcopy(), deterministic=False)

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._evaluation_action_selector(state, begin_of_episode=begin_of_episode)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        return None if self._is_buffer(env_or_buffer) else self._explorer_builder(self._env_info, self._config, self)

    def _setup_policy_training(self, env_or_buffer):
        is_offline = isinstance(env_or_buffer, ReplayBuffer)
        if is_offline:
            policy_trainer_config = MT.policy_trainers.XQLForwardPolicyTrainerConfig(
                beta=self._config.policy_temperature,
                unroll_steps=self._config.pi_unroll_steps,
                burn_in_steps=self._config.pi_burn_in_steps,
                reset_on_terminal=self._config.pi_reset_rnn_on_terminal)
            policy_trainer = MT.policy_trainers.XQLForwardPolicyTrainer(
                models=self._pi,
                solvers={self._pi.scope_name: self._pi_solver},
                q_functions=self._target_q_functions,
                v_function=self._v_function,
                env_info=self._env_info,
                config=policy_trainer_config)
            return policy_trainer
        else:
            raise NotImplementedError

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_config = MT.q_value_trainers.VTargetedQTrainerConfig(
            loss_type='huber',
            reduction_method='mean',
            num_steps=self._config.num_steps,
            huber_delta=20.0,
            q_loss_scalar=0.5,
            unroll_steps=self._config.q_unroll_steps,
            burn_in_steps=self._config.q_burn_in_steps,
            reset_on_terminal=self._config.q_reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.VTargetedQTrainer(
            train_functions=self._train_q_functions,
            solvers=self._train_q_solvers,
            target_functions=self._v_function,
            env_info=self._env_info,
            config=q_function_trainer_config)
        return q_function_trainer

    def _setup_v_function_training(self, env_or_buffer):
        is_offline = isinstance(env_or_buffer, ReplayBuffer)

        v_function_trainer_config = MT.v_value_trainers.XQLVTrainerConfig(
            reduction_method='mean',
            beta=self._config.value_temperature,
            unroll_steps=self._config.v_unroll_steps,
            burn_in_steps=self._config.v_burn_in_steps,
            reset_on_terminal=self._config.v_reset_rnn_on_terminal)

        v_function_trainer = MT.v_value_trainers.XQLVTrainer(
            train_functions=self._v_function,
            solvers={self._v_function.scope_name: self._v_solver},
            target_functions=self._target_q_functions,
            target_policy=None if is_offline else self._pi,
            env_info=self._env_info,
            config=v_function_trainer_config)
        return v_function_trainer

    def _run_online_training_iteration(self, env):
        raise NotImplementedError

    def _run_offline_training_iteration(self, buffer):
        self._xql_training(buffer)

    def _xql_training(self, replay_buffer):
        pi_steps = self._config.pi_burn_in_steps + self._config.pi_unroll_steps
        q_steps = self._config.num_steps + self._config.q_burn_in_steps + self._config.q_unroll_steps - 1
        v_steps = self._config.v_burn_in_steps + self._config.v_unroll_steps
        num_steps = max(pi_steps, max(q_steps, v_steps))
        experiences_tuple, info = replay_buffer.sample(self._config.batch_size, num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = (experiences_tuple, )
        assert len(experiences_tuple) == num_steps

        batch = None
        for experiences in reversed(experiences_tuple):
            (s, a, r, non_terminal, s_next, rnn_states_dict, *_) = marshal_experiences(experiences)
            rnn_states = rnn_states_dict['rnn_states'] if 'rnn_states' in rnn_states_dict else {}
            batch = TrainingBatch(batch_size=self._config.batch_size,
                                  s_current=s,
                                  a_current=a,
                                  gamma=self._config.gamma,
                                  reward=r,
                                  non_terminal=non_terminal,
                                  s_next=s_next,
                                  weight=info['weights'],
                                  next_step_batch=batch,
                                  rnn_states=rnn_states)

        self._v_function_trainer_state = self._v_function_trainer.train(batch)
        self._policy_trainer_state = self._policy_trainer.train(batch)
        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            sync_model(q, target_q, tau=self._config.tau)

        td_errors = self._q_function_trainer_state['td_errors']
        replay_buffer.update_priorities(td_errors)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _exploration_action_selector(self, s, *, begin_of_episode=False):
        return self._exploration_actor(s, begin_of_episode=begin_of_episode)

    def _models(self):
        models = [*self._train_q_functions, self._v_function, self._pi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers[self._pi.scope_name] = self._pi_solver
        solvers[self._v_function.scope_name] = self._v_solver
        solvers.update(self._train_q_solvers)
        return solvers

    @classmethod
    def is_rnn_supported(self):
        return True

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return not env_info.is_discrete_action_env() and not env_info.is_tuple_action_env()

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(XQL, self).latest_iteration_state
        if hasattr(self, '_policy_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_loss': float(self._policy_trainer_state['pi_loss'])})
        if hasattr(self, '_v_function_trainer_state'):
            latest_iteration_state['scalar'].update({'v_loss': float(self._v_function_trainer_state['v_loss'])})
        if hasattr(self, '_q_function_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': float(self._q_function_trainer_state['q_loss'])})
            latest_iteration_state['histogram'].update(
                {'td_errors': self._q_function_trainer_state['td_errors'].flatten()})
        return latest_iteration_state

    @property
    def trainers(self):
        return {"q_function": self._q_function_trainer,
                "v_function": self._v_function_trainer,
                "policy": self._policy_trainer}
