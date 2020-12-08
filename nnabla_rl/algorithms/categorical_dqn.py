import nnabla as nn

import numpy as np

from dataclasses import dataclass

import nnabla.solvers as NS

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import C51ValueDistributionFunction, ValueDistributionFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def default_value_distribution_builder(scope_name, env_info, algorithm_params, **kwargs):
    return C51ValueDistributionFunction(scope_name,
                                        env_info.state_shape,
                                        env_info.action_dim,
                                        algorithm_params.num_atoms,
                                        algorithm_params.v_min,
                                        algorithm_params.v_max)


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


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


class CategoricalDQN(Algorithm):
    '''Categorical DQN algorithm implementation.

    This class implements the Categorical DQN algorithm
    proposed by M. Bellemare, et al. in the paper: "A Distributional Perspective on Reinfocement Learning"
    For detail see: https://arxiv.org/pdf/1707.06887.pdf
    '''

    def __init__(self, env_or_env_info,
                 value_distribution_builder=default_value_distribution_builder,
                 replay_buffer_builder=default_replay_buffer_builder,
                 params=CategoricalDQNParam()):
        super(CategoricalDQN, self).__init__(env_or_env_info, params=params)
        if not self._env_info.is_discrete_action_env():
            raise ValueError('{} only supports discrete action environment'.format(self.__name__))

        def solver_builder():
            return NS.Adam(alpha=self._params.learning_rate, eps=1e-2 / self._params.batch_size)
        self._atom_p = value_distribution_builder('atom_p_train', self._env_info, self._params)
        self._atom_p_solver = {self._atom_p.scope_name: solver_builder()}
        self._target_atom_p = self._atom_p.deepcopy('target_atom_p_train')
        assert isinstance(self._atom_p, ValueDistributionFunction)
        assert isinstance(self._target_atom_p, ValueDistributionFunction)

        self._replay_buffer = replay_buffer_builder(params.replay_buffer_size)

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
            gamma=self._params.gamma,
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
        model_trainer.setup_training(self._atom_p, self._atom_p_solver, training)

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
        marshalled_experiences = marshall_experiences(experiences)

        kwargs = {}
        kwargs['weights'] = info['weights']
        errors = self._model_trainer.train(marshalled_experiences, **kwargs)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _greedy_action_selector(self, s):
        s_eval_var = nn.Variable.from_numpy_array(np.expand_dims(s, axis=0))

        q_function = self._atom_p.as_q_function()
        with nn.auto_forward():
            a_greedy = q_function.argmax_q(s_eval_var)
        return np.squeeze(a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _models(self):
        models = {}
        models[self._atom_p.scope_name] = self._atom_p
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._atom_p_solver)
        return solvers
