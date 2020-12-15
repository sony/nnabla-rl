import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass

import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.model_trainers.model_trainer import Training
from nnabla_rl.models import TD3QFunction, TD3Policy, QFunction, DeterministicPolicy
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT


def default_critic_builder(scope_name, env_info, algorithm_params, **kwargs):
    target_policy = kwargs.get('target_policy')
    return TD3QFunction(scope_name, env_info.state_dim, env_info.action_dim, optimal_policy=target_policy)


def default_actor_builder(scope_name, env_info, algorithm_params, **kwargs):
    max_action_value = float(env_info.action_space.high[0])
    return TD3Policy(scope_name, env_info.state_dim, env_info.action_dim, max_action_value=max_action_value)


@dataclass
class TD3Param(AlgorithmParam):
    d: int = 2
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    exploration_noise_sigma: float = 0.1
    train_action_noise_sigma: float = 0.2
    train_action_noise_abs: float = 0.5
    batch_size: int = 100
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        if not (0 < self.d):
            raise ValueError('d must be greater than 0')
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')


class TD3(Algorithm):
    def __init__(self, env_or_env_info,
                 critic_builder=default_critic_builder,
                 actor_builder=default_actor_builder,
                 params=TD3Param()):
        super(TD3, self).__init__(env_or_env_info, params=params)

        def policy_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._pi = actor_builder(scope_name="pi", env_info=self._env_info, algorithm_params=self._params)
        self._pi_solver = {self._pi.scope_name: policy_solver_builder()}
        self._target_pi = self._pi.deepcopy('target_' + self._pi.scope_name)
        assert isinstance(self._pi, DeterministicPolicy)
        assert isinstance(self._target_pi, DeterministicPolicy)

        self._q1 = critic_builder(scope_name="q1", env_info=self._env_info, algorithm_params=self._params)
        self._q2 = critic_builder(scope_name="q2", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._q1, QFunction)
        assert isinstance(self._q2, QFunction)
        train_q_functions = [self._q1, self._q2]

        def q_function_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._train_q_functions = train_q_functions
        self._train_q_solvers = {q.scope_name: q_function_solver_builder() for q in train_q_functions}
        self._target_q_functions = [q.deepcopy('target_' + q.scope_name) for q in train_q_functions]

        self._replay_buffer = ReplayBuffer(capacity=params.replay_buffer_size)

    @eval_api
    def compute_eval_action(self, state):
        action, _ = self._compute_greedy_action(state)
        return action

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_params = EE.GaussianExplorerParam(
            warmup_random_steps=self._params.start_timesteps,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False,
            action_clip_low=self._env_info.action_space.low,
            action_clip_high=self._env_info.action_space.high,
            sigma=self._params.exploration_noise_sigma
        )
        explorer = EE.GaussianExplorer(policy_action_selector=self._compute_greedy_action,
                                       env_info=self._env_info,
                                       params=explorer_params)
        return explorer

    def _setup_q_function_training(self, env_or_buffer):
        # training input/loss variables
        q_function_trainer_params = MT.q_value_trainers.SquaredTDQFunctionTrainerParam(
            reduction_method='mean',
            grad_clip=None)
        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            env_info=self._env_info,
            params=q_function_trainer_params)

        training = MT.q_value_trainings.TD3Training(
            train_functions=self._train_q_functions,
            target_functions=self._target_q_functions,
            target_policy=self._target_pi,
            train_action_noise_sigma=self._params.train_action_noise_sigma,
            train_action_noise_abs=self._params.train_action_noise_abs)
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._train_q_functions,
            dst_models=self._target_q_functions,
            target_update_frequency=self._params.d,
            tau=self._params.tau)
        q_function_trainer.setup_training(self._train_q_functions, self._train_q_solvers, training)
        for q, target_q in zip(self._train_q_functions, self._target_q_functions):
            copy_network_parameters(q.get_parameters(), target_q.get_parameters())
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.DPGPolicyTrainerParam()
        policy_trainer = MT.policy_trainers.DPGPolicyTrainer(env_info=self._env_info,
                                                             params=policy_trainer_params,
                                                             q_function=self._q1)

        training = Training()
        # Update policy everytime when train is called. Because train is called every self._params.d iteration
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._pi,
            dst_models=self._target_pi,
            target_update_frequency=1,
            tau=self._params.tau)
        policy_trainer.setup_training(self._pi, self._pi_solver, training)
        copy_network_parameters(self._pi.get_parameters(), self._target_pi.get_parameters(), 1.0)

        return policy_trainer

    def _run_online_training_iteration(self, env):
        experiences = self._environment_explorer.step(env)
        self._replay_buffer.append_all(experiences)

        if self._params.start_timesteps < self.iteration_num:
            self._td3_training(self._replay_buffer)

    def _run_offline_training_iteration(self, buffer):
        self._td3_training(buffer)

    def _td3_training(self, replay_buffer):
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

        if self.iteration_num % self._params.d == 0:
            # Optimize actor
            self._policy_trainer.train(batch)

    def _compute_greedy_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            self._eval_action = self._pi.pi(self._eval_state_var)
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    def _models(self):
        models = {}
        models[self._q1.scope_name] = self._q1
        models[self._q2.scope_name] = self._q2
        models[self._pi.scope_name] = self._pi
        models[self._target_pi.scope_name] = self._target_pi
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._pi_solver)
        solvers.update(self._train_q_solvers)
        return solvers
