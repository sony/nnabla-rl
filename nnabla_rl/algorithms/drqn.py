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
from typing import Union

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import eval_api
from nnabla_rl.algorithms.dqn import DQN, DefaultReplayBufferBuilder, DQNConfig
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import DRQNQFunction, QFunction
from nnabla_rl.utils import context
from nnabla_rl.utils.data import add_batch_dimension, marshal_experiences, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables, sync_model


@dataclass
class DRQNConfig(DQNConfig):
    """
    List of configurations for DRQN algorithm. Most of the configs are inherited from DQNConfig

    Args:
        unroll_steps (int): Number of steps to unroll recurrent layer during training. Defaults to 10.
    """

    unroll_steps: int = 10
    # Overriding some configurations from original DQNConfig
    learning_rate: float = 0.1
    replay_buffer_size: int = 400000

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        super().__post_init__()
        self._assert_positive(self.unroll_steps, 'unroll_steps')


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DRQNConfig,
                     **kwargs) -> nn.solver.Solver:
        decay: float = 0.95
        solver = NS.Adadelta(lr=algorithm_config.learning_rate, decay=decay)
        return solver


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DRQNConfig,
                    **kwargs) -> QFunction:
        return DRQNQFunction(scope_name, env_info.action_dim)


class DefaultExplorerBuilder(ExplorerBuilder):
    def build_explorer(self,  # type: ignore[override]
                       env_info: EnvironmentInfo,
                       algorithm_config: DRQNConfig,
                       algorithm: "DRQN",
                       **kwargs) -> EnvironmentExplorer:
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            warmup_random_steps=algorithm_config.start_timesteps,
            initial_step_num=algorithm.iteration_num,
            initial_epsilon=algorithm_config.initial_epsilon,
            final_epsilon=algorithm_config.final_epsilon,
            max_explore_steps=algorithm_config.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=algorithm._exploration_action_selector,
            random_action_selector=algorithm._random_action_selector,
            env_info=env_info,
            config=explorer_config)
        return explorer


class _GreedyActionSelector(object):
    def __init__(self, env_info, q_function: QFunction):
        self._env_info = env_info
        self._q = q_function.shallowcopy()

    @eval_api
    def __call__(self, s, *, begin_of_episode=False):
        s = add_batch_dimension(s)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = create_variable(1, self._env_info.state_shape)
            if self._q.is_recurrent():
                self._rnn_internal_states = create_variables(1, self._q.internal_state_shapes())
                self._q.set_internal_states(self._rnn_internal_states)
            self._a_greedy = self._q.argmax_q(self._eval_state_var)
        if self._q.is_recurrent() and begin_of_episode:
            self._q.reset_internal_states()
        set_data_to_variable(self._eval_state_var, s)
        if self._q.is_recurrent():
            prev_rnn_states = self._q.get_internal_states()
            for key in self._rnn_internal_states.keys():
                # copy internal states of previous iteration
                self._rnn_internal_states[key].d = prev_rnn_states[key].d
        self._a_greedy.forward()
        # No need to save internal states
        return np.squeeze(self._a_greedy.d, axis=0), {}


class DRQN(DQN):
    '''DRQN algorithm.

    This class implements the Bootstrapped random update version of Deep Recurrent Q-Network (DRQN) algorithm.
    proposed by M. Hausknecht, et al. in the paper: "Deep Recurrent Q-Learning for Partially Observable MDPs"
    For details see: https://arxiv.org/pdf/1507.06527.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DRQNConfig <nnabla_rl.algorithms.drqn.DRQNConfig>`):
            the parameter for DRQN training
        q_func_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of q function model
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of q function solver
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer
        explorer_builder (:py:class:`ExplorerBuilder <nnabla_rl.builders.ExplorerBuilder>`):
            builder of environment explorer
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _evaluation_actor: _GreedyActionSelector
    _exploration_actor: _GreedyActionSelector
    _config: DRQNConfig

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DRQNConfig = DRQNConfig(),
                 q_func_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder(),
                 explorer_builder: ExplorerBuilder = DefaultExplorerBuilder()):
        super(DRQN, self).__init__(env_or_env_info,
                                   config=config,
                                   q_func_builder=q_func_builder,
                                   q_solver_builder=q_solver_builder,
                                   replay_buffer_builder=replay_buffer_builder,
                                   explorer_builder=explorer_builder)

        self._evaluation_actor = _GreedyActionSelector(self._env_info, self._q)
        self._exploration_actor = _GreedyActionSelector(self._env_info, self._q)

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            (action, _), _ = epsilon_greedy_action_selection(state,
                                                             self._evaluation_action_selector,
                                                             self._random_action_selector,
                                                             epsilon=self._config.test_epsilon,
                                                             begin_of_episode=begin_of_episode)
            return action

    @classmethod
    def is_rnn_supported(self):
        return True

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value_trainers.DQNQTrainerConfig(
            num_steps=self._config.num_steps,
            reduction_method='mean',
            grad_clip=self._config.grad_clip,
            unroll_steps=self._config.unroll_steps,
            reset_on_terminal=False)

        q_function_trainer = MT.q_value_trainers.DQNQTrainer(
            train_functions=self._q,
            solvers={self._q.scope_name: self._q_solver},
            target_function=self._target_q,
            env_info=self._env_info,
            config=trainer_config)
        sync_model(self._q, self._target_q)
        return q_function_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._config.start_timesteps < self.iteration_num:
            if self.iteration_num % self._config.learner_update_frequency == 0:
                self._drqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._drqn_training(buffer)

    def _drqn_training(self, replay_buffer):
        num_steps = self._config.num_steps + self._config.unroll_steps - 1
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
        if self.iteration_num % self._config.target_update_frequency == 0:
            sync_model(self._q, self._target_q)

        td_errors = self._q_function_trainer_state['td_errors']
        replay_buffer.update_priorities(td_errors)

    def _evaluation_action_selector(self, s, *, begin_of_episode=False):
        return self._evaluation_actor(s, begin_of_episode=begin_of_episode)

    def _exploration_action_selector(self, s, *, begin_of_episode=False):
        return self._exploration_actor(s, begin_of_episode=begin_of_episode)

    def _random_action_selector(self, s, *, begin_of_episode=False):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}
