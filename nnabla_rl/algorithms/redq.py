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
from typing import Union

import gym

import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithms.sac import (SAC, DefaultExplorerBuilder, DefaultPolicyBuilder, DefaultQFunctionBuilder,
                                      DefaultReplayBufferBuilder, DefaultSolverBuilder, SACConfig)
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import QFunction, StochasticPolicy
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class REDQConfig(SACConfig):
    '''REDQConfig
    List of configurations for REDQ algorithm

    Args:
        G (int): Number of update-to-data ratio. Defaults to 20.
        M (int): Size of subset M. Defaults to 2.
        N (int): Number of q functions of an ensemble. Defaults to 10.
    '''
    # override timesteps
    start_timesteps: int = 5000

    # REDQ specific parameters
    G: int = 20
    M: int = 2
    N: int = 10

    def __post_init__(self):
        '''__post_init__
        Check set values are in valid range.
        '''
        super().__post_init__()
        self._assert_positive(self.G, 'G')
        self._assert_positive(self.N, 'N')
        self._assert_positive(self.M, 'M')


class REDQ(SAC):
    '''Randomized Ensembled Double Q-learning (REDQ) algorithm implementation.

    This class implements the Randomized Ensembled Double Q-learning (REDQ) algorithm
    proposed by X. Chen, et al. in the paper: "Randomized Ensembled Double Q-learning: Learning Fast Without a Model"
    For detail see: https://arxiv.org/abs/2101.05982.
    Note that our implementation is implemented on top of SAC. (Same as the original paper).

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`REDQConfig <nnabla_rl.algorithms.redq.REDQConfig>`): configuration of the REDQ algorithm
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
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: REDQConfig

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: REDQConfig = REDQConfig(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 temperature_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super().__init__(env_or_env_info,
                         config=config,
                         q_function_builder=q_function_builder,
                         q_solver_builder=q_solver_builder,
                         policy_builder=policy_builder,
                         policy_solver_builder=policy_solver_builder,
                         temperature_solver_builder=temperature_solver_builder,
                         replay_buffer_builder=replay_buffer_builder,
                         explorer_builder=explorer_builder)

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_config = MT.q_value_trainers.REDQQTrainerConfig(
            reduction_method='mean',
            grad_clip=None,
            M=self._config.M,
            num_steps=self._config.num_steps,
            unroll_steps=self._config.critic_unroll_steps,
            burn_in_steps=self._config.critic_burn_in_steps,
            reset_on_terminal=self._config.critic_reset_rnn_on_terminal)

        q_function_trainer = MT.q_value_trainers.REDQQTrainer(
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

    def _run_online_training_iteration(self, env):
        for _ in range(self._config.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._config.gradient_steps):
            self._run_gradient_step(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._redq_training(buffer)

    def _run_environment_step(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)

    def _run_gradient_step(self, replay_buffer):
        if self._config.start_timesteps < self.iteration_num:
            self._redq_training(replay_buffer)

    def _redq_training(self, replay_buffer):
        actor_steps = self._config.actor_burn_in_steps + self._config.actor_unroll_steps
        critic_steps = self._config.num_steps + self._config.critic_burn_in_steps + self._config.critic_unroll_steps - 1
        num_steps = max(actor_steps, critic_steps)

        for _ in range(self._config.G):
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
            for q, target_q in zip(self._train_q_functions, self._target_q_functions):
                sync_model(q, target_q, tau=self._config.tau)
            td_errors = self._q_function_trainer_state['td_errors']
            replay_buffer.update_priorities(td_errors)

        self._policy_trainer_state = self._policy_trainer.train(batch)

    def _build_q_functions(self, q_function_builder):
        q_functions = []
        for i in range(self._config.N):
            q = q_function_builder(scope_name=f"q{i+1}", env_info=self._env_info, algorithm_config=self._config)
            q_functions.append(q)
        return q_functions
