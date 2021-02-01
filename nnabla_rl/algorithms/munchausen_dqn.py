import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import gym
import numpy as np

from typing import cast, Union

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.builders import QFunctionBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import DQNQFunction, QFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


@dataclass
class MunchausenDQNParam(AlgorithmParam):
    gamma: float = 0.99
    batch_size: int = 32
    # optimizer
    learning_rate: float = 0.00005
    # network update
    learner_update_frequency: float = 4
    target_update_frequency: float = 10000
    # buffers
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    # explore
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    max_explore_steps: int = 1000000
    # munchausen dqn training parameters
    entropy_temperature: float = 0.03
    munchausen_scaling_term: float = 0.9
    clipping_value: float = -1

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.learner_update_frequency, 'learner_update_frequency')
        self._assert_positive(self.target_update_frequency, 'target_update_frequency')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_smaller_than(self.start_timesteps, self.replay_buffer_size, 'start_timesteps')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_between(self.initial_epsilon, 0.0, 1.0, 'initial_epsilon')
        self._assert_between(self.final_epsilon, 0.0, 1.0, 'final_epsilon')
        self._assert_between(self.test_epsilon, 0.0, 1.0, 'test_epsilon')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_negative(self.clipping_value, 'clipping_value')


class DefaultQFunctionBuilder(QFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: MunchausenDQNParam,
                    **kwargs) -> QFunction:
        return DQNQFunction(scope_name, env_info.action_dim)


class DefaultQSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_params: MunchausenDQNParam,
                     **kwargs) -> nn.solvers.Solver:
        assert isinstance(algorithm_params, MunchausenDQNParam)
        return NS.Adam(algorithm_params.learning_rate, eps=1e-2 / algorithm_params.batch_size)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_params: MunchausenDQNParam,
                            **kwargs) -> ReplayBuffer:
        assert isinstance(algorithm_params, MunchausenDQNParam)
        return ReplayBuffer(capacity=algorithm_params.replay_buffer_size)


class MunchausenDQN(Algorithm):
    '''Munchausen-DQN algorithm implementation.

    This class implements the Munchausen-DQN (Munchausen Deep Q Network) algorithm
    proposed by N. Vieillard, et al. in the paper: "Munchausen Reinforcement Learning"
    For detail see: https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf
    '''

    _params: MunchausenDQNParam
    _q: QFunction
    _target_q: QFunction
    _q_solver: nn.solver.Solver
    _replay_buffer: ReplayBuffer

    _environment_explorer: EnvironmentExplorer
    _quantile_function_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _a_greedy: nn.Variable

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 params: MunchausenDQNParam = MunchausenDQNParam(),
                 q_func_builder: QFunctionBuilder = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultQSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(MunchausenDQN, self).__init__(env_or_env_info, params=params)

        if not self._env_info.is_discrete_action_env():
            raise ValueError('Invalid env_info. Action space of MunchausenDQN must be {}' .format(gym.spaces.Discrete))

        self._q = q_func_builder(scope_name='q', env_info=self._env_info, algorithm_params=self._params)
        self._q_solver = q_solver_builder(env_info=self._env_info, algorithm_params=self._params)
        self._target_q = cast(QFunction, self._q.deepcopy('target_' + self._q.scope_name))

        self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_params=self._params)

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
            reduction_method='mean',
            q_loss_scalar=0.5,
            grad_clip=(-1.0, 1.0))

        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            params=trainer_params)

        target_update_frequency = self._params.target_update_frequency // self._params.learner_update_frequency
        training = MT.q_value_trainings.MunchausenRLTraining(train_function=self._q,
                                                             target_function=self._target_q,
                                                             tau=self._params.entropy_temperature,
                                                             alpha=self._params.munchausen_scaling_term,
                                                             clip_min=self._params.clipping_value,
                                                             clip_max=0.0)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._q,
            dst_models=self._target_q,
            target_update_frequency=target_update_frequency,
            tau=1.0)
        q_function_trainer.setup_training(self._q, {self._q.scope_name: self._q_solver}, training)
        copy_network_parameters(self._q.get_parameters(), self._target_q.get_parameters())
        return q_function_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._params.start_timesteps < self.iteration_num:
            if self.iteration_num % self._params.learner_update_frequency == 0:
                self._m_dqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._m_dqn_training(buffer)

    def _greedy_action_selector(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._a_greedy = self._q.argmax_q(self._eval_state_var)
        self._eval_state_var.d = s
        self._a_greedy.forward()
        return np.squeeze(self._a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _m_dqn_training(self, replay_buffer):
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
        solvers[self._q.scope_name] = self._q_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        return latest_iteration_state
