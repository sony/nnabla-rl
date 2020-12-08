import nnabla as nn

import numpy as np

from dataclasses import dataclass

import nnabla.solvers as NS

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import IQNQuantileFunction, StateActionQuantileFunction
from nnabla_rl.environment_explorers.epsilon_greedy_explorer import epsilon_greedy_action_selection
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def risk_neutral_measure(tau):
    return tau


def default_quantile_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    risk_measure_function = kwargs['risk_measure_function']
    return IQNQuantileFunction(scope_name,
                               env_info.state_shape,
                               env_info.action_dim,
                               algorithm_params.embedding_dim,
                               K=algorithm_params.K,
                               risk_measure_function=risk_measure_function)


def default_replay_buffer_builder(capacity):
    return ReplayBuffer(capacity=capacity)


@dataclass
class IQNParam(AlgorithmParam):
    batch_size: int = 32
    gamma: float = 0.99
    start_timesteps: int = 50000
    replay_buffer_size: int = 1000000
    learner_update_frequency: int = 4
    target_update_frequency: int = 10000
    max_explore_steps: int = 1000000
    learning_rate: float = 0.00005
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    test_epsilon: float = 0.001
    N: int = 64
    N_prime: int = 64
    K: int = 32
    kappa: float = 1.0
    embedding_dim: int = 64

    def __post_init__(self):
        '''__post_init__

        Check that set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.replay_buffer_size, 'replay_buffer_size')
        self._assert_positive(self.learner_update_frequency, 'learner_update_frequency')
        self._assert_positive(self.target_update_frequency, 'target_update_frequency')
        self._assert_positive(self.max_explore_steps, 'max_explore_steps')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.initial_epsilon, 'initial_epsilon')
        self._assert_positive(self.final_epsilon, 'final_epsilon')
        self._assert_positive(self.test_epsilon, 'test_epsilon')
        self._assert_positive(self.N, 'N')
        self._assert_positive(self.N_prime, 'N_prime')
        self._assert_positive(self.K, 'K')
        self._assert_positive(self.kappa, 'kappa')
        self._assert_positive(self.embedding_dim, 'embedding_dim')


class IQN(Algorithm):
    '''Implicit Quantile Network algorithm implementation.

    This class implements the Implicit Quantile Network (IQN) algorithm
    proposed by W. Dabney, et al. in the paper: "Implicit Quantile Networks for Distributional Reinforcement Learning"
    For detail see: https://arxiv.org/pdf/1806.06923.pdf
    '''

    def __init__(self, env_or_env_info,
                 quantile_function_builder=default_quantile_function_builder,
                 risk_measure_function=risk_neutral_measure,
                 params=IQNParam(),
                 replay_buffer_builder=default_replay_buffer_builder):
        super(IQN, self).__init__(env_or_env_info, params=params)

        if not self._env_info.is_discrete_action_env():
            raise ValueError('{} only supports discrete action environment'.format(self.__name__))

        kwargs = {}
        kwargs['risk_measure_function'] = risk_measure_function
        self._quantile_function = quantile_function_builder('quantile_function', self._env_info, self._params, **kwargs)
        self._target_quantile_function = self._quantile_function.deepcopy('target_quantile_function')
        assert isinstance(self._quantile_function, StateActionQuantileFunction)
        assert isinstance(self._target_quantile_function, StateActionQuantileFunction)

        def solver_builder():
            return NS.Adam(alpha=self._params.learning_rate, eps=1e-2 / self._params.batch_size)
        self._quantile_function_solver = {self._quantile_function.scope_name: solver_builder()}

        self._replay_buffer = replay_buffer_builder(capacity=params.replay_buffer_size)

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._quantile_function_trainer = self._setup_quantile_function_training(env_or_buffer)

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

    def _setup_quantile_function_training(self, env_or_buffer):
        trainer_params = MT.q_value_trainers.IQNQuantileFunctionTrainerParam(
            gamma=self._params.gamma,
            N=self._params.N,
            N_prime=self._params.N_prime,
            K=self._params.K,
            kappa=self._params.kappa)

        quantile_function_trainer = MT.q_value_trainers.IQNQuantileFunctionTrainer(
            self._env_info,
            params=trainer_params)

        target_update_frequency = self._params.target_update_frequency / self._params.learner_update_frequency
        training = MT.q_value_trainings.DQNTraining(
            train_function=self._quantile_function,
            target_function=self._target_quantile_function)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._quantile_function,
            dst_models=self._target_quantile_function,
            target_update_frequency=target_update_frequency,
            tau=1.0)
        quantile_function_trainer.setup_training(self._quantile_function, self._quantile_function_solver, training)

        # NOTE: Copy initial parameters after setting up the training
        # Because the parameter is created after training graph construction
        copy_network_parameters(self._quantile_function.get_parameters(),
                                self._target_quantile_function.get_parameters())

        return quantile_function_trainer

    def compute_eval_action(self, state):
        (action, _), _ = epsilon_greedy_action_selection(state,
                                                         self._greedy_action_selector,
                                                         self._random_action_selector,
                                                         epsilon=self._params.test_epsilon)
        return action

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)
        if self._params.start_timesteps < self.iteration_num:
            if self.iteration_num % self._params.learner_update_frequency == 0:
                self._iqn_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._iqn_training(buffer)

    def _iqn_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._params.batch_size)
        marshalled_experiences = marshall_experiences(experiences)

        self._quantile_function_trainer.train(marshalled_experiences)

    def _greedy_action_selector(self, s):
        s_eval_var = nn.Variable.from_numpy_array(np.expand_dims(s, axis=0))
        with nn.auto_forward():
            q_function = self._quantile_function.as_q_function()
            a_greedy = q_function.argmax_q(s_eval_var)
        return np.squeeze(a_greedy.d, axis=0), {}

    def _random_action_selector(self, s):
        action = self._env_info.action_space.sample()
        return np.asarray(action).reshape((1, )), {}

    def _models(self):
        models = {}
        models[self._quantile_function.scope_name] = self._quantile_function
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._quantile_function_solver)
        return solvers
