# Copyright 2020,2021 Sony Corporation.
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
from typing import Any, Dict, Union

import gym

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import _DeterministicPolicyActionSelector
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import DeterministicPolicy, QFunction, TD3Policy, TD3QFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class DDPGConfig(AlgorithmConfig):
    '''DDPGConfig
    List of configurations for DDPG algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size(int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.\
            Defaults to 10000.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        exploration_noise_sigma (float): standard deviation of gaussian exploration noise. Defaults to 0.1.
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
    '''

    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    tau: float = 0.005
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    exploration_noise_sigma: float = 0.1
    num_steps: int = 1

    # rnn model support
    actor_unroll_steps: int = 1
    actor_burn_in_steps: int = 0
    actor_reset_rnn_on_terminal: bool = True

    critic_unroll_steps: int = 1
    critic_burn_in_steps: int = 0
    critic_reset_rnn_on_terminal: bool = True

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.exploration_noise_sigma, 'exploration_noise_sigma')

        self._assert_positive(self.critic_unroll_steps, 'critic_unroll_steps')
        self._assert_positive_or_zero(self.critic_burn_in_steps, 'critic_burn_in_steps')
        self._assert_positive(self.actor_unroll_steps, 'actor_unroll_steps')
        self._assert_positive_or_zero(self.actor_burn_in_steps, 'actor_burn_in_steps')


class DefaultCriticBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DDPGConfig,
                    **kwargs) -> QFunction:
        target_policy = kwargs.get('target_policy')
        return TD3QFunction(scope_name, optimal_policy=target_policy)


class DefaultActorBuilder(ModelBuilder[DeterministicPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DDPGConfig,
                    **kwargs) -> DeterministicPolicy:
        max_action_value = float(env_info.action_high[0])
        return TD3Policy(scope_name, env_info.action_dim, max_action_value=max_action_value)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DDPGConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: DDPGConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: DDPGConfig,
                       algorithm: "DDPG",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.GaussianExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            initial_step_num=algorithm.iteration_num,
            timelimit_as_terminal=False,
            action_clip_low=env_info.action_low,
            action_clip_high=env_info.action_high,
            sigma=algorithm_config.exploration_noise_sigma
        )
        explorer = EE.GaussianExplorer(policy_action_selector=algorithm._exploration_action_selector,
                                       env_info=env_info,
                                       config=explorer_config)
        return explorer


class DDPG(Algorithm):
    '''Deep Deterministic Policy Gradient (DDPG) algorithm.

    This class implements the modified version of the Deep Deterministic Policy Gradient (DDPG) algorithm
    proposed by T. P.  Lillicrap, et al. in the paper: "Continuous control with deep reinforcement learning"
    For details see: https://arxiv.org/abs/1509.02971
    We use gaussian noise instead of Ornstein-Uhlenbeck process to explore in the environment.
    The effectiveness of using gaussian noise for DDPG is reported in the paper:
    "Addressing Funciton Approximaiton Error in Actor-Critic Methods". see https://arxiv.org/abs/1802.09477

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DDPGConfig <nnabla_rl.algorithms.ddpg.DDPGConfig>`):
            configuration of the DDPG algorithm
        critic_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of critic models
        critic_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of critic solvers
        actor_builder (:py:class:`ModelBuilder[DeterministicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of actor models
        actor_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of actor solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DDPGConfig
    _q: QFunction
    _q_solver: nn.solver.Solver
    _target_q: QFunction
    _pi: DeterministicPolicy
    _pi_solver: nn.solver.Solver
    _target_pi: DeterministicPolicy
    _replay_buffer: ReplayBuffer
    _explorer_builder: ExplorerBuilder
    _environment_explorer: EnvironmentExplorer
    _q_function_trainer: ModelTrainer
    _policy_trainer: ModelTrainer

    _policy_trainer_state: Dict[str, Any]
    _q_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DDPGConfig = DDPGConfig(),
                 critic_builder: ModelBuilder[QFunction] = DefaultCriticBuilder(),
                 critic_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 actor_builder: ModelBuilder[DeterministicPolicy] = DefaultActorBuilder(),
                 actor_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(DDPG, self).__init__(env_or_env_info, config=config)

        self._explorer_builder = explorer_builder

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._q = critic_builder(scope_name="q", env_info=self._env_info, algorithm_config=self._config)
            self._q_solver = critic_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._target_q = self._q.deepcopy('target_' + self._q.scope_name)

            self._pi = actor_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
            self._pi_solver = actor_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._target_pi = self._pi.deepcopy("target_" + self._pi.scope_name)

            self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)

        self._evaluation_actor = _DeterministicPolicyActionSelector(self._env_info, self._pi.shallowcopy())
        self._exploration_actor = _DeterministicPolicyActionSelector(self._env_info, self._pi.shallowcopy())

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            action, _ = self._evaluation_action_selector(state, begin_of_episode=begin_of_episode)
            return action

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        return None if self._is_buffer(env_or_buffer) else self._explorer_builder(self._env_info, self._config, self)

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_config = MT.q_value_trainers.DDPGQTrainerConfig(
            reduction_method='mean',
            grad_clip=None,
            num_steps=self._config.num_steps,
            unroll_steps=self._config.critic_unroll_steps,
            burn_in_steps=self._config.critic_burn_in_steps,
            reset_on_terminal=self._config.critic_reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.DDPGQTrainer(
            train_functions=self._q,
            solvers={self._q.scope_name: self._q_solver},
            target_functions=self._target_q,
            target_policy=self._target_pi,
            env_info=self._env_info,
            config=q_function_trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.DPGPolicyTrainerConfig(
            unroll_steps=self._config.actor_unroll_steps,
            burn_in_steps=self._config.actor_burn_in_steps,
            reset_on_terminal=self._config.actor_reset_rnn_on_terminal)

        policy_trainer = MT.policy_trainers.DPGPolicyTrainer(
            models=self._pi,
            solvers={self._pi.scope_name: self._pi_solver},
            q_function=self._q,
            env_info=self._env_info,
            config=policy_trainer_config)
        sync_model(self._pi, self._target_pi, tau=1.0)
        return policy_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._config.start_timesteps < self.iteration_num:
            self._ddpg_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._ddpg_training(buffer)

    def _ddpg_training(self, replay_buffer):
        actor_steps = self._config.actor_burn_in_steps + self._config.actor_unroll_steps
        critic_steps = self._config.num_steps + self._config.critic_burn_in_steps + self._config.critic_unroll_steps - 1
        num_steps = max(actor_steps, critic_steps)
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

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        sync_model(self._q, self._target_q, tau=self._config.tau)

        self._policy_trainer_state = self._policy_trainer.train(batch)
        sync_model(self._pi, self._target_pi, tau=self._config.tau)

        td_errors = self._q_function_trainer_state['td_errors']
        replay_buffer.update_priorities(td_errors)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _exploration_action_selector(self, s, *, begin_of_episode=False):
        return self._exploration_actor(s, begin_of_episode=begin_of_episode)

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._pi.scope_name] = self._pi_solver
        solvers[self._q.scope_name] = self._q_solver
        return solvers

    @classmethod
    def is_rnn_supported(self):
        return True

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return not env_info.is_discrete_action_env()

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(DDPG, self).latest_iteration_state
        if hasattr(self, '_policy_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_loss': float(self._policy_trainer_state['pi_loss'])})
        if hasattr(self, '_q_function_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': float(self._q_function_trainer_state['q_loss'])})
            latest_iteration_state['histogram'].update(
                {'td_errors': self._q_function_trainer_state['td_errors'].flatten()})
        return latest_iteration_state

    @property
    def trainers(self):
        return {"q_function": self._q_function_trainer, "policy": self._policy_trainer}
