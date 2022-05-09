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
from typing import Any, Dict, List, Optional, Union

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
from nnabla_rl.models import QFunction, SACPolicy, SACQFunction, SACVFunction, StochasticPolicy, VFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class DEMMESACConfig(AlgorithmConfig):
    '''DEMMESACConfig
    List of configurations for DEMMESAC algorithm.

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
        reward_scalar (float): Reward scaling factor. Obtained reward will be multiplied by this value. Defaults to 5.0.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        num_rr_steps (int): number of steps for N-step Q_rr targets. Defaults to 1.
        num_re_steps (int): number of steps for N-step Q_re targets. Defaults to 1.
        target_update_interval (float): the interval of target v function parameter's update. Defaults to 1.
        pi_t_unroll_steps (int): Number of steps to unroll policy's (pi_t) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        pi_e_unroll_steps (int): Number of steps to unroll policy's (pi_e) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        pi_t_burn_in_steps (int): Number of burn-in steps to initiaze policy's (pi_t) recurrent layer states
            during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        pi_e_burn_in_steps (int): Number of burn-in steps to initiaze policy's (pi_e) recurrent layer states
            during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        pi_t_reset_rnn_on_terminal (bool): Reset policy's (pi_t) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        pi_e_reset_rnn_on_terminal (bool): Reset policy's (pi_e) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        q_rr_unroll_steps (int): Number of steps to unroll q-function's (q_rr) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        q_re_unroll_steps (int): Number of steps to unroll q-function's (q_re) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        q_rr_burn_in_steps (int): Number of burn-in steps to initiaze q-function's (q_rr) recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        q_re_burn_in_steps (int): Number of burn-in steps to initiaze q-function's (q_re) recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        q_rr_reset_rnn_on_terminal (bool): Reset q-function's (q_rr) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        q_re_reset_rnn_on_terminal (bool): Reset q-function's (q_re) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        v_rr_unroll_steps (int): Number of steps to unroll v-function's (v_rr) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        v_re_unroll_steps (int): Number of steps to unroll v-function's (v_re) tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        v_rr_burn_in_steps (int): Number of burn-in steps to initiaze v-function's (v_rr) recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        v_re_burn_in_steps (int): Number of burn-in steps to initiaze v-function's (v_re) recurrent layer states\
            during training. This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        v_rr_reset_rnn_on_terminal (bool): Reset v-function's (v_rr) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        v_re_reset_rnn_on_terminal (bool): Reset v-function's (v_re) recurrent internal states to zero during training\
            if episode ends. This flag does not take effect if given model is not an RNN model.\
            Defaults to False.
        alpha_pi (Optional[float]): If None, will use reward_scalar to scale the reward.
            Otherwise 1/alpha_pi will be used to scale the reward. Defaults to None.
        alpha_q (float): Temperature value for negative entropy term. Defaults to 1.0.
    '''

    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    batch_size: int = 256
    tau: float = 0.005
    environment_steps: int = 1
    gradient_steps: int = 1
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    target_update_interval: int = 1
    num_rr_steps: int = 1
    num_re_steps: int = 1

    # temperature values
    reward_scalar: float = 5.0
    alpha_pi: Optional[float] = None
    alpha_q: float = 1.0

    # rnn model support
    pi_t_unroll_steps: int = 1
    pi_t_burn_in_steps: int = 0
    pi_t_reset_rnn_on_terminal: bool = True

    pi_e_unroll_steps: int = 1
    pi_e_burn_in_steps: int = 0
    pi_e_reset_rnn_on_terminal: bool = True

    q_rr_unroll_steps: int = 1
    q_rr_burn_in_steps: int = 0
    q_rr_reset_rnn_on_terminal: bool = True

    q_re_unroll_steps: int = 1
    q_re_burn_in_steps: int = 0
    q_re_reset_rnn_on_terminal: bool = True

    v_rr_unroll_steps: int = 1
    v_rr_burn_in_steps: int = 0
    v_rr_reset_rnn_on_terminal: bool = True

    v_re_unroll_steps: int = 1
    v_re_burn_in_steps: int = 0
    v_re_reset_rnn_on_terminal: bool = True

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.gradient_steps, 'gradient_steps')
        self._assert_positive(self.environment_steps, 'environment_steps')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.target_update_interval, 'target_update_interval')
        self._assert_positive(self.num_rr_steps, 'num_rr_steps')
        self._assert_positive(self.num_re_steps, 'num_re_steps')

        self._assert_positive(self.pi_t_unroll_steps, 'pi_t_unroll_steps')
        self._assert_positive_or_zero(self.pi_t_burn_in_steps, 'pi_t_burn_in_steps')
        self._assert_positive(self.pi_e_unroll_steps, 'pi_e_unroll_steps')
        self._assert_positive_or_zero(self.pi_e_burn_in_steps, 'pi_e_burn_in_steps')
        self._assert_positive(self.q_rr_unroll_steps, 'q_rr_unroll_steps')
        self._assert_positive_or_zero(self.q_rr_burn_in_steps, 'q_rr_burn_in_steps')
        self._assert_positive(self.q_re_unroll_steps, 'q_re_unroll_steps')
        self._assert_positive_or_zero(self.q_re_burn_in_steps, 'q_re_burn_in_steps')
        self._assert_positive(self.v_rr_unroll_steps, 'v_rr_unroll_steps')
        self._assert_positive_or_zero(self.v_rr_burn_in_steps, 'v_rr_burn_in_steps')
        self._assert_positive(self.v_re_unroll_steps, 'v_re_unroll_steps')
        self._assert_positive_or_zero(self.v_re_burn_in_steps, 'v_re_burn_in_steps')

        if self.alpha_pi is not None:
            # Recompute with alpha_pi
            self.reward_scalar = 1 / self.alpha_pi


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DEMMESACConfig,
                    **kwargs) -> VFunction:
        return SACVFunction(scope_name)


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DEMMESACConfig,
                    **kwargs) -> QFunction:
        return SACQFunction(scope_name)


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DEMMESACConfig,
                    **kwargs) -> StochasticPolicy:
        return SACPolicy(scope_name, env_info.action_dim)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DEMMESACConfig,
                     **kwargs) -> nn.solver.Solver:
        assert isinstance(algorithm_config, DEMMESACConfig)
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: DEMMESACConfig,
                            **kwargs) -> ReplayBuffer:
        assert isinstance(algorithm_config, DEMMESACConfig)
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: DEMMESACConfig,
                       algorithm: "DEMMESAC",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.RawPolicyExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            reward_scalar=algorithm_config.reward_scalar,
            initial_step_num=algorithm.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=algorithm._exploration_action_selector,
                                        env_info=env_info,
                                        config=explorer_config)
        return explorer


class DEMMESAC(Algorithm):
    '''DisEntangled Max-Min Entropy Soft Actor-Critic (DEMME-SAC) algorithm.

    This class implements the disentangled version of max-min Soft Actor Critic (SAC) algorithm proposed
    by S. Han, et al. in the paper: "A Max-Min Entropy Framework for Reinforcement Learning"
    For detail see: https://arxiv.org/abs/2106.10517

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DEMMESACConfig <nnabla_rl.algorithms.demme_sac.DEMMESACConfig>`):
            configuration of the DEMMESAC algorithm
        v_rr_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of reward v function models
        v_rr_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of reward v function solvers
        v_re_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of entropy v function models
        v_re_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of entropyv function solvers
        q_rr_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of reward q function models
        q_rr_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of reward q function solvers
        q_re_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of entropy q function models
        q_re_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of entropy q function solvers
        pi_t_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of target policy models
        pi_t_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of target policy solvers
        pi_e_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of pure exploration policy models
        pi_e_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of pure exploration policy solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DEMMESACConfig
    _pi_t: StochasticPolicy
    _pi_e: StochasticPolicy
    _v_rr: VFunction
    _v_re: VFunction
    _v_rr_solver: nn.solver.Solver
    _v_re_solver: nn.solver.Solver
    _target_v_rr: VFunction
    _target_v_re: VFunction
    _q_rr1: QFunction
    _q_rr2: QFunction
    _q_re1: QFunction
    _q_re2: QFunction
    _train_q_rr_functions: List[QFunction]
    _train_q_rr_solvers: Dict[str, nn.solver.Solver]
    _train_q_re_functions: List[QFunction]
    _train_q_re_solvers: Dict[str, nn.solver.Solver]
    _replay_buffer: ReplayBuffer
    _explorer_builder: ExplorerBuilder
    _environment_explorer: EnvironmentExplorer
    _pi_t_trainer: ModelTrainer
    _pi_e_trainer: ModelTrainer
    _q_rr_trainer: ModelTrainer
    _q_re_trainer: ModelTrainer
    _v_rr_trainer: ModelTrainer
    _v_re_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _eval_deterministic_action: nn.Variable
    _eval_probabilistic_action: nn.Variable

    _pi_t_trainer_state: Dict[str, Any]
    _pi_e_trainer_state: Dict[str, Any]
    _q_rr_trainer_state: Dict[str, Any]
    _q_re_trainer_state: Dict[str, Any]
    _v_rr_trainer_state: Dict[str, Any]
    _v_re_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DEMMESACConfig = DEMMESACConfig(),
                 v_rr_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_rr_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 v_re_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_re_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 q_rr_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_rr_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 q_re_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_re_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 pi_t_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 pi_t_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 pi_e_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 pi_e_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(DEMMESAC, self).__init__(env_or_env_info, config=config)

        self._explorer_builder = explorer_builder

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._v_rr = v_rr_function_builder(
                scope_name="v_rr", env_info=self._env_info, algorithm_config=self._config)
            self._v_rr_solver = v_rr_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._v_re = v_re_function_builder(
                scope_name="v_re", env_info=self._env_info, algorithm_config=self._config)
            self._v_re_solver = v_re_solver_builder(env_info=self._env_info, algorithm_config=self._config)

            self._target_v_rr = self._v_rr.deepcopy('target_' + self._v_rr.scope_name)
            self._target_v_re = self._v_re.deepcopy('target_' + self._v_re.scope_name)

            self._q_rr1 = q_rr_function_builder(
                scope_name="q_rr1", env_info=self._env_info, algorithm_config=self._config)
            self._q_rr2 = q_rr_function_builder(
                scope_name="q_rr2", env_info=self._env_info, algorithm_config=self._config)

            self._train_q_rr_functions = [self._q_rr1, self._q_rr2]
            self._train_q_rr_solvers = {}
            for q in self._train_q_rr_functions:
                self._train_q_rr_solvers[q.scope_name] = q_rr_solver_builder(
                    env_info=self._env_info, algorithm_config=self._config)

            self._q_re1 = q_re_function_builder(
                scope_name="q_re1", env_info=self._env_info, algorithm_config=self._config)
            self._q_re2 = q_re_function_builder(
                scope_name="q_re2", env_info=self._env_info, algorithm_config=self._config)

            self._train_q_re_functions = [self._q_re1, self._q_re2]
            self._train_q_re_solvers = {}
            for q in self._train_q_re_functions:
                self._train_q_re_solvers[q.scope_name] = q_re_solver_builder(
                    env_info=self._env_info, algorithm_config=self._config)

            self._pi_t = pi_t_builder(scope_name="pi_t", env_info=self._env_info, algorithm_config=self._config)
            self._pi_t_solver = pi_t_solver_builder(env_info=self._env_info, algorithm_config=self._config)

            self._pi_e = pi_e_builder(scope_name="pi_e", env_info=self._env_info, algorithm_config=self._config)
            self._pi_e_solver = pi_e_solver_builder(env_info=self._env_info, algorithm_config=self._config)

            self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)

        self._evaluation_actor = _StochasticPolicyActionSelector(
            self._env_info, self._pi_t.shallowcopy(), deterministic=True)
        self._exploration_actor = _StochasticPolicyActionSelector(
            self._env_info, self._pi_t.shallowcopy(), deterministic=False)

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._evaluation_action_selector(state, begin_of_episode=begin_of_episode)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._pi_t_trainer = self._setup_pi_t_training(env_or_buffer)
        self._pi_e_trainer = self._setup_pi_e_training(env_or_buffer)
        self._q_rr_trainer = self._setup_q_rr_training(env_or_buffer)
        self._q_re_trainer = self._setup_q_re_training(env_or_buffer)
        self._v_rr_trainer = self._setup_v_rr_training(env_or_buffer)
        self._v_re_trainer = self._setup_v_re_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        return None if self._is_buffer(env_or_buffer) else self._explorer_builder(self._env_info, self._config, self)

    def _setup_pi_t_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.DEMMEPolicyTrainerConfig(
            unroll_steps=self._config.pi_t_unroll_steps,
            burn_in_steps=self._config.pi_t_burn_in_steps,
            reset_on_terminal=self._config.pi_t_reset_rnn_on_terminal)
        policy_trainer = MT.policy_trainers.DEMMEPolicyTrainer(
            models=self._pi_t,
            solvers={self._pi_t.scope_name: self._pi_t_solver},
            q_rr_functions=self._train_q_rr_functions,
            q_re_functions=self._train_q_re_functions,
            env_info=self._env_info,
            config=policy_trainer_config)
        return policy_trainer

    def _setup_pi_e_training(self, env_or_buffer):
        # NOTE: Fix temperature to 1.0. Because This version of SAC adjusts it by scaling the reward
        policy_trainer_config = MT.policy_trainers.SoftPolicyTrainerConfig(
            fixed_temperature=True,
            unroll_steps=self._config.pi_e_unroll_steps,
            burn_in_steps=self._config.pi_e_burn_in_steps,
            reset_on_terminal=self._config.pi_e_reset_rnn_on_terminal)
        temperature = MT.policy_trainers.soft_policy_trainer.AdjustableTemperature(
            scope_name='temperature',
            initial_value=1.0)
        policy_trainer = MT.policy_trainers.SoftPolicyTrainer(
            models=self._pi_e,
            solvers={self._pi_e.scope_name: self._pi_e_solver},
            q_functions=self._train_q_re_functions,
            temperature=temperature,
            temperature_solver=None,
            env_info=self._env_info,
            config=policy_trainer_config)
        return policy_trainer

    def _setup_q_rr_training(self, env_or_buffer):
        q_rr_trainer_param = MT.q_value_trainers.VTargetedQTrainerConfig(
            reduction_method='mean',
            q_loss_scalar=0.5,
            num_steps=self._config.num_rr_steps,
            unroll_steps=self._config.q_rr_unroll_steps,
            burn_in_steps=self._config.q_rr_burn_in_steps,
            reset_on_terminal=self._config.q_rr_reset_rnn_on_terminal)
        q_rr_trainer = MT.q_value_trainers.VTargetedQTrainer(
            train_functions=self._train_q_rr_functions,
            solvers=self._train_q_rr_solvers,
            target_functions=self._target_v_rr,
            env_info=self._env_info,
            config=q_rr_trainer_param)
        return q_rr_trainer

    def _setup_q_re_training(self, env_or_buffer):
        q_re_trainer_param = MT.q_value_trainers.VTargetedQTrainerConfig(
            reduction_method='mean',
            q_loss_scalar=0.5,
            num_steps=self._config.num_re_steps,
            unroll_steps=self._config.q_re_unroll_steps,
            burn_in_steps=self._config.q_re_burn_in_steps,
            reset_on_terminal=self._config.q_re_reset_rnn_on_terminal,
            pure_exploration=True)
        q_re_trainer = MT.q_value_trainers.VTargetedQTrainer(
            train_functions=self._train_q_re_functions,
            solvers=self._train_q_re_solvers,
            target_functions=self._target_v_re,
            env_info=self._env_info,
            config=q_re_trainer_param)
        return q_re_trainer

    def _setup_v_rr_training(self, env_or_buffer):
        v_rr_trainer_config = MT.v_value_trainers.DEMMEVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=0.5,
            unroll_steps=self._config.v_rr_unroll_steps,
            burn_in_steps=self._config.v_rr_burn_in_steps,
            reset_on_terminal=self._config.v_rr_reset_rnn_on_terminal)
        v_rr_trainer = MT.v_value_trainers.DEMMEVTrainer(
            train_functions=self._v_rr,
            solvers={self._v_rr.scope_name: self._v_rr_solver},
            target_functions=self._train_q_rr_functions,  # Set training q_rr as target
            target_policy=self._pi_t,
            env_info=self._env_info,
            config=v_rr_trainer_config)
        sync_model(self._v_rr, self._target_v_rr, 1.0)

        return v_rr_trainer

    def _setup_v_re_training(self, env_or_buffer):
        alpha_q = MT.policy_trainers.soft_policy_trainer.AdjustableTemperature(
            scope_name='alpha_q',
            initial_value=self._config.alpha_q)
        v_re_trainer_config = MT.v_value_trainers.MMEVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=0.5,
            unroll_steps=self._config.v_re_unroll_steps,
            burn_in_steps=self._config.v_re_burn_in_steps,
            reset_on_terminal=self._config.v_re_reset_rnn_on_terminal)
        v_re_trainer = MT.v_value_trainers.MMEVTrainer(
            train_functions=self._v_re,
            temperature=alpha_q,
            solvers={self._v_re.scope_name: self._v_re_solver},
            target_functions=self._train_q_re_functions,  # Set training q_re as target
            target_policy=self._pi_e,
            env_info=self._env_info,
            config=v_re_trainer_config)
        sync_model(self._v_re, self._target_v_re, 1.0)

        return v_re_trainer

    def _run_online_training_iteration(self, env):
        for _ in range(self._config.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._config.gradient_steps):
            self._run_gradient_step(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._demme_training(buffer)

    def _run_environment_step(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)

    def _run_gradient_step(self, replay_buffer):
        if self._config.start_timesteps < self.iteration_num:
            self._demme_training(replay_buffer)

    def _demme_training(self, replay_buffer):
        pi_t_steps = self._config.pi_t_burn_in_steps + self._config.pi_t_unroll_steps
        pi_e_steps = self._config.pi_e_burn_in_steps + self._config.pi_e_unroll_steps
        q_rr_steps = self._config.num_rr_steps + self._config.q_rr_burn_in_steps + self._config.q_rr_unroll_steps - 1
        q_re_steps = self._config.num_re_steps + self._config.q_re_burn_in_steps + self._config.q_re_unroll_steps - 1
        v_rr_steps = self._config.v_rr_burn_in_steps + self._config.v_rr_unroll_steps
        v_re_steps = self._config.v_re_burn_in_steps + self._config.v_re_unroll_steps
        pi_steps = max(pi_t_steps, pi_e_steps)
        q_steps = max(q_rr_steps, q_re_steps)
        v_steps = max(v_rr_steps, v_re_steps)
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

        # Train in the order of v -> q -> policy
        self._v_rr_trainer_state = self._v_rr_trainer.train(batch)
        self._v_re_trainer_state = self._v_re_trainer.train(batch)
        self._q_rr_trainer_state = self._q_rr_trainer.train(batch)
        self._q_re_trainer_state = self._q_re_trainer.train(batch)
        if self.iteration_num % self._config.target_update_interval == 0:
            sync_model(self._v_rr, self._target_v_rr, tau=self._config.tau)
            sync_model(self._v_re, self._target_v_re, tau=self._config.tau)
        self._pi_t_trainer_state = self._pi_t_trainer.train(batch)
        self._pi_e_trainer_state = self._pi_e_trainer.train(batch)

        # Use q_rr's td error
        td_errors = self._q_rr_trainer_state['td_errors']
        replay_buffer.update_priorities(td_errors)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _exploration_action_selector(self, s, *, begin_of_episode=False):
        return self._exploration_actor(s, begin_of_episode=begin_of_episode)

    def _models(self):
        models = [self._v_rr, self._v_re, self._target_v_rr, self._target_v_re,
                  self._q_rr1, self._q_rr2, self._q_re1, self._q_re2,
                  self._pi_t, self._pi_e]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._train_q_rr_solvers)
        solvers.update(self._train_q_re_solvers)
        solvers[self._v_rr.scope_name] = self._v_rr_solver
        solvers[self._v_re.scope_name] = self._v_re_solver
        solvers[self._pi_t.scope_name] = self._pi_t_solver
        solvers[self._pi_e.scope_name] = self._pi_e_solver
        return solvers

    @classmethod
    def is_rnn_supported(cls):
        return True

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return not env_info.is_discrete_action_env()

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(DEMMESAC, self).latest_iteration_state
        if hasattr(self, '_pi_t_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_t_loss': float(self._pi_t_trainer_state['pi_loss'])})
        if hasattr(self, '_pi_e_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_e_loss': float(self._pi_e_trainer_state['pi_loss'])})
        if hasattr(self, '_v_rr_trainer_state'):
            latest_iteration_state['scalar'].update({'v_re_loss': float(self._v_re_trainer_state['v_loss'])})
        if hasattr(self, '_v_re_trainer_state'):
            latest_iteration_state['scalar'].update({'v_rr_loss': float(self._v_rr_trainer_state['v_loss'])})
        if hasattr(self, '_q_rr_trainer_state'):
            latest_iteration_state['scalar'].update({'q_rr_loss': float(self._q_rr_trainer_state['q_loss'])})
            latest_iteration_state['histogram'].update(
                {'q_rr_td_errors': self._q_rr_trainer_state['td_errors'].flatten()})
        if hasattr(self, '_q_re_trainer_state'):
            latest_iteration_state['scalar'].update({'q_re_loss': float(self._q_re_trainer_state['q_loss'])})
            latest_iteration_state['histogram'].update(
                {'q_re_td_errors': self._q_re_trainer_state['td_errors'].flatten()})
        return latest_iteration_state

    @property
    def trainers(self):
        return {
            "v_rr": self._v_rr_trainer,
            "v_re": self._v_re_trainer,
            "q_rr": self._q_rr_trainer,
            "q_re": self._q_re_trainer,
            "pi_t": self._pi_t_trainer,
            "pi_e": self._pi_e_trainer,
        }
