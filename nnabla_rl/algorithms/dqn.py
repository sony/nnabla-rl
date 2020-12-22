import nnabla as nn
import nnabla.solvers as NS

import warnings
from dataclasses import dataclass

import gym
import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import DQNQFunction, QFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def default_q_func_builder(scope_name, env_info, algorithm_params, **kwargs):
    return DQNQFunction(scope_name, env_info.action_dim)


def default_q_solver_builder(params):
    try:
        solver = NS.RMSpropGraves(
            lr=params.learning_rate, decay=params.decay,
            momentum=params.momentum, eps=params.min_squared_gradient)
    except AttributeError:
        warnings.warn("Instead of RMSpropGraves, use Adam as a Solver, \
            Please check learning rate. It might be needed to tune it")
        solver = NS.Adam(params.learning_rate)
    return solver


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


@dataclass
class DQNParam(AlgorithmParam):
    gamma: float = 0.99
    batch_size: int = 32
    # optimizer
    learning_rate: float = 2.5e-4
    # this decay is equivalent to 'gradient momentum' and 'squared gradient momentum' of the nature paper
    decay: float = 0.95
    momentum: float = 0.0
    min_squared_gradient: float = 0.01
    # network update
    learner_update_frequency: float = 4
    target_update_frequency: float = 10000
    # buffers
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    # explore
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.1
    test_epsilon: float = 0.05
    max_explore_steps: int = 1000000

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 <= self.batch_size):
            raise ValueError('batch size must not be negative')
        if not (0 <= self.learning_rate):
            raise ValueError('learning rate must not be negative')
        if not (0 <= self.decay):
            raise ValueError('decay must not be negative')
        if not (0 <= self.min_squared_gradient):
            raise ValueError('min_squared_gradient must not be negative')
        if not (0 <= self.learner_update_frequency):
            raise ValueError('learner update frequency must not be negative')
        if not (0 <= self.target_update_frequency):
            raise ValueError('target update frequency must not be negative')
        if self.start_timesteps is not None:
            if not (0 <= self.start_timesteps):
                raise ValueError('start timesteps must not be negative')
        if (self.start_timesteps > self.replay_buffer_size):
            raise ValueError('start timesteps should be smaller than \
                replay buffer size')
        if not (0 <= self.replay_buffer_size):
            raise ValueError('replay buffer size must not be negative')
        if not ((0.0 <= self.initial_epsilon) & (self.initial_epsilon <= 1.0)):
            raise ValueError('initial epsilon must lie between [0.0, 1.0]')
        if not ((0.0 <= self.final_epsilon) & (self.final_epsilon <= 1.0)):
            raise ValueError('final epsilon must lie between [0.0, 1.0]')
        if not ((0.0 <= self.test_epsilon) & (self.test_epsilon <= 1.0)):
            raise ValueError('test epsilon must lie between [0.0, 1.0]')
        if not (0 <= self.max_explore_steps):
            raise ValueError('max explore step must not be negative')


class DQN(Algorithm):
    def __init__(self, env_or_env_info,
                 q_func_builder=default_q_func_builder,
                 params=DQNParam(),
                 replay_buffer_builder=default_replay_buffer_builder):
        super(DQN, self).__init__(env_or_env_info, params=params)

        if not self._env_info.is_discrete_action_env():
            raise ValueError('Invalid env_info. Action space of DQN must be {}' .format(gym.spaces.Discrete))

        def solver_builder():
            return default_q_solver_builder(self._params)
        self._q = q_func_builder(scope_name='q', env_info=self._env_info, algorithm_params=self._params)
        self._q_solver = {self._q.scope_name: solver_builder()}
        self._target_q = self._q.deepcopy('target_' + self._q.scope_name)
        assert isinstance(self._q, QFunction)
        assert isinstance(self._target_q, QFunction)

        self._replay_buffer = replay_buffer_builder(capacity=params.replay_buffer_size)

    @eval_api
    def compute_eval_action(self, s):
        (action, _), _ = epsilon_greedy_action_selection(s,
                                                         self._greedy_action_selector,
                                                         self._random_action_selector,
                                                         epsilon=self._params.test_epsilon)
        return action

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None

        explorer_params = EE.LinearDecayEpsilonGreedyExplorerParam(
            warmup_random_steps=self._params.start_timesteps,
            initial_step_num=self.iteration_num,
            initial_epsilon=self._params.initial_epsilon,
            final_epsilon=self._params.final_epsilon,
            max_explore_steps=self._params.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=self._greedy_action_selector,
            random_action_selector=self._random_action_selector,
            env_info=self._env_info,
            params=explorer_params)
        return explorer

    def _setup_q_function_training(self, env_or_buffer):
        trainer_params = MT.q_value_trainers.SquaredTDQFunctionTrainerParam(
            reduction_method='sum',
            grad_clip=(-1.0, 1.0))

        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            params=trainer_params)

        target_update_frequency = self._params.target_update_frequency / self._params.learner_update_frequency
        training = MT.q_value_trainings.DQNTraining(train_function=self._q, target_function=self._target_q)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._q,
            dst_models=self._target_q,
            target_update_frequency=target_update_frequency,
            tau=1.0)
        q_function_trainer.setup_training(self._q, self._q_solver, training)
        copy_network_parameters(self._q.get_parameters(), self._target_q.get_parameters())
        return q_function_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._params.start_timesteps < self.iteration_num:
            if self.iteration_num % self._params.learner_update_frequency == 0:
                self._dqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._dqn_training(buffer)

    def _greedy_action_selector(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._a_greedy = self._q.max_q(self._eval_state_var)
        self._eval_state_var.d = s
        self._a_greedy.forward()
        return np.squeeze(self._a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _dqn_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._params.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshall_experiences(experiences)
        batch = TrainingBatch(batch_size=self._params.batch_size,
                              s_current=s,
                              a_current=a,
                              gamma=self._params.gamma,
                              reward=r,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=info['weights'])

        errors = self._q_function_trainer.train(batch)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _models(self):
        models = {}
        models[self._q.scope_name] = self._q
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._q_solver)
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        return latest_iteration_state
