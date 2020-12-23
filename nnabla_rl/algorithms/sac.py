import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import SACQFunction, SACPolicy, QFunction, StochasticPolicy
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def default_q_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return SACQFunction(scope_name)


def default_policy_builder(scope_name, env_info, algorithm_params, **kwargs):
    return SACPolicy(scope_name, env_info.action_dim)


@dataclass
class SACParam(AlgorithmParam):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    environment_steps: int = 1
    gradient_steps: int = 1
    target_entropy: float = None
    initial_temperature: float = None
    fix_temperature: bool = False
    batch_size: int = 256
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 < self.gradient_steps):
            raise ValueError('gradient steps must be greater than 0')
        if not (0 < self.environment_steps):
            raise ValueError('environment steps must be greater than 0')
        if (self.initial_temperature is not None):
            if (self.initial_temperature <= 0.0):
                raise ValueError('temperature must be greater than 0')
        if not (0 <= self.start_timesteps):
            raise ValueError('start_timesteps must not be negative')


class SAC(Algorithm):
    '''Soft Actor-Critic (SAC) algorithm implementation.

    This class implements the extended version of Soft Actor Critic (SAC) algorithm
    proposed by T. Haarnoja, et al. in the paper: "Soft Actor-Critic Algorithms and Applications"
    For detail see: https://arxiv.org/pdf/1812.05905.pdf

    This algorithm is slightly differs from the implementation of Soft Actor-Critic algorithm presented
    also by T. Haarnoja, et al. in the following paper:  https://arxiv.org/pdf/1801.01290.pdf

    The temperature parameter is adjusted automatically instead of providing reward scalar as a
    hyper parameter.

    '''

    def __init__(self, env_or_env_info,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 params=SACParam()):
        super(SAC, self).__init__(env_or_env_info, params=params)

        def q_function_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._q1 = q_function_builder(scope_name="q1", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._q1, QFunction)
        self._q2 = q_function_builder(scope_name="q2", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._q2, QFunction)
        train_q_functions = [self._q1, self._q2]
        self._train_q_functions = train_q_functions
        self._train_q_solvers = {q.scope_name: q_function_solver_builder() for q in train_q_functions}
        self._target_q_functions = [q.deepcopy('target_' + q.scope_name) for q in train_q_functions]

        def policy_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._pi, StochasticPolicy)
        self._pi_solver = {self._pi.scope_name: policy_solver_builder()}

        def temperature_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._temperature = MT.policy_trainers.soft_policy_trainer.AdjustableTemperature(
            scope_name='temperature',
            initial_value=self._params.initial_temperature)
        if not self._params.fix_temperature:
            self._temperature_solver = temperature_solver_builder()
        else:
            self._temperature_solver = None

        self._replay_buffer = ReplayBuffer(capacity=params.replay_buffer_size)

    @eval_api
    def compute_eval_action(self, state):
        action, _ = self._compute_greedy_action(state, deterministic=True)
        return action

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_params = EE.RawPolicyExplorerParam(
            warmup_random_steps=self._params.start_timesteps,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_greedy_action,
                                        env_info=self._env_info,
                                        params=explorer_params)
        return explorer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.SoftPolicyTrainerParam(
            fixed_temperature=self._params.fix_temperature,
            target_entropy=self._params.target_entropy)
        policy_trainer = MT.policy_trainers.SoftPolicyTrainer(
            env_info=self._env_info,
            params=policy_trainer_params,
            temperature=self._temperature,
            temperature_solver=self._temperature_solver,
            q_functions=[self._q1, self._q2])

        training = MT.model_trainer.Training()
        policy_trainer.setup_training(self._pi, self._pi_solver, training)
        return policy_trainer

    def _setup_q_function_training(self, env_or_buffer):
        # training input/loss variables
        q_function_trainer_params = MT.q_value_trainers.SquaredTDQFunctionTrainerParam(
            reduction_method='mean',
            grad_clip=None)

        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            params=q_function_trainer_params)

        training = MT.q_value_trainings.SoftQTraining(
            train_functions=self._train_q_functions,
            target_functions=self._target_q_functions,
            target_policy=self._pi,
            temperature=self._policy_trainer.get_temperature())
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._train_q_functions,
            dst_models=self._target_q_functions,
            target_update_frequency=1,
            tau=self._params.tau)
        q_function_trainer.setup_training(self._train_q_functions, self._train_q_solvers, training)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            copy_network_parameters(q.get_parameters(), target_q.get_parameters())
        return q_function_trainer

    def _run_online_training_iteration(self, env):
        for _ in range(self._params.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._params.gradient_steps):
            self._run_gradient_step(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._sac_training(buffer)

    def _run_environment_step(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)

    def _run_gradient_step(self, replay_buffer):
        if self._params.start_timesteps < self.iteration_num:
            self._sac_training(replay_buffer)

    def _sac_training(self, replay_buffer):
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
        self._policy_trainer.train(batch)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _compute_greedy_action(self, s, deterministic=False):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._pi.pi(self._eval_state_var)
            if deterministic:
                self._eval_action = distribution.choose_probable()
            else:
                self._eval_action = distribution.sample()
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    def _models(self):
        models = [self._q1, self._q2, self._pi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._pi_solver)
        solvers.update(self._train_q_solvers)
        if self._temperature_solver is not None:
            solvers.update({self._temperature.scope_name: self._temperature_solver})
        return solvers
