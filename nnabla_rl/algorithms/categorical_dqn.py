import nnabla as nn

import numpy as np

from dataclasses import dataclass

import nnabla.solvers as NS

import gym
from typing import cast, Union

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.builders import ValueDistributionFunctionBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import C51ValueDistributionFunction, ValueDistributionFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


@dataclass
class CategoricalDQNParam(AlgorithmParam):
    batch_size: int = 32
    gamma: float = 0.99
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    max_explore_steps: int = 1000000
    learning_rate: float = 0.00025
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51


class DefaultValueDistFunctionBuilder(ValueDistributionFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: CategoricalDQNParam,
                    **kwargs) -> ValueDistributionFunction:
        return C51ValueDistributionFunction(scope_name,
                                            env_info.action_dim,
                                            algorithm_params.num_atoms,
                                            algorithm_params.v_min,
                                            algorithm_params.v_max)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_params: CategoricalDQNParam,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_params.replay_buffer_size)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_params: CategoricalDQNParam,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_params.learning_rate, eps=1e-2 / algorithm_params.batch_size)


class CategoricalDQN(Algorithm):
    '''Categorical DQN algorithm implementation.

    This class implements the Categorical DQN algorithm
    proposed by M. Bellemare, et al. in the paper: "A Distributional Perspective on Reinfocement Learning"
    For detail see: https://arxiv.org/pdf/1707.06887.pdf
    '''

    _params: CategoricalDQNParam
    _atom_p: ValueDistributionFunction
    _atom_p_solver: nn.solver.Solver
    _target_atom_p: ValueDistributionFunction
    _replay_buffer: ReplayBuffer
    _environment_explorer: EnvironmentExplorer
    _model_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _a_greedy: nn.Variable

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 params: CategoricalDQNParam = CategoricalDQNParam(),
                 value_distribution_builder: ValueDistributionFunctionBuilder = DefaultValueDistFunctionBuilder(),
                 value_distribution_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(CategoricalDQN, self).__init__(env_or_env_info, params=params)
        if not self._env_info.is_discrete_action_env():
            raise ValueError('{} only supports discrete action environment'.format(self.__name__))

        self._atom_p = value_distribution_builder('atom_p_train', self._env_info, self._params)
        self._atom_p_solver = value_distribution_solver_builder(self._env_info, self._params)
        self._target_atom_p = cast(ValueDistributionFunction, self._atom_p.deepcopy('target_atom_p_train'))

        self._replay_buffer = replay_buffer_builder(self._env_info, self._params)

    @eval_api
    def compute_eval_action(self, state):
        (action, _), _ = epsilon_greedy_action_selection(state,
                                                         self._greedy_action_selector,
                                                         self._random_action_selector,
                                                         epsilon=self._params.test_epsilon)
        return action

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._model_trainer = self._setup_value_distribution_function_training(env_or_buffer)

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

    def _setup_value_distribution_function_training(self, env_or_buffer):
        trainer_params = MT.q_value_trainers.C51ValueDistributionFunctionTrainerParam(
            v_min=self._params.v_min,
            v_max=self._params.v_max,
            num_atoms=self._params.num_atoms)

        model_trainer = MT.q_value_trainers.C51ValueDistributionFunctionTrainer(
            self._env_info,
            params=trainer_params)

        target_update_frequency = self._params.target_update_frequency / self._params.learner_update_frequency
        training = MT.q_value_trainings.DQNTraining(
            train_function=self._atom_p,
            target_function=self._target_atom_p)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._atom_p,
            dst_models=self._target_atom_p,
            target_update_frequency=target_update_frequency,
            tau=1.0)
        model_trainer.setup_training(self._atom_p, {self._atom_p.scope_name: self._atom_p_solver}, training)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        copy_network_parameters(self._atom_p.get_parameters(),
                                self._target_atom_p.get_parameters())
        return model_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._params.start_timesteps < self.iteration_num:
            if self.iteration_num % self._params.learner_update_frequency == 0:
                self._categorical_dqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._categorical_dqn_training(buffer)

    def _categorical_dqn_training(self, replay_buffer):
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

        errors = self._model_trainer.train(batch)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _greedy_action_selector(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            q_function = self._atom_p.as_q_function()
            self._a_greedy = q_function.argmax_q(self._eval_state_var)
        self._eval_state_var.d = s
        self._a_greedy.forward()
        return np.squeeze(self._a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _models(self):
        models = {}
        models[self._atom_p.scope_name] = self._atom_p
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._atom_p.scope_name] = self._atom_p_solver
        return solvers
