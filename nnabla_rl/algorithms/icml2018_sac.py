from dataclasses import dataclass

import numpy as np

import nnabla_rl.model_trainers as MT
import nnabla_rl.environment_explorers as EE
import nnabla as nn
import nnabla.solvers as NS
from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.utils.copy import copy_network_parameters
from nnabla_rl.models import SACVFunction, SACQFunction, SACPolicy, VFunction, QFunction, StochasticPolicy
from nnabla_rl.model_trainers.model_trainer import TrainingBatch


def default_v_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return SACVFunction(scope_name, env_info.state_dim)


def default_q_function_builder(scope_name, env_info, algorithm_params, **kwargs):
    return SACQFunction(scope_name)


def default_policy_builder(scope_name, env_info, algorithm_params, **kwargs):
    return SACPolicy(scope_name, env_info.action_dim)


@dataclass
class ICML2018SACParam(AlgorithmParam):
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3.0*1e-4
    environment_steps: int = 1
    gradient_steps: int = 1
    reward_scalar: float = 5.0
    batch_size: int = 256
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000
    target_update_interval: int = 1

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.gradient_steps, 'gradient_steps')
        self._assert_positive(self.environment_steps, 'environment_steps')
        self._assert_positive(self.start_timesteps, 'start_timesteps')
        self._assert_positive(self.target_update_interval,
                              'target_update_interval')


class ICML2018SAC(Algorithm):
    '''Soft Actor-Critic (SAC) algorithm implementation.

    This class implements the ICML2018 version of Soft Actor Critic (SAC) algorithm proposed by T. Haarnoja, et al.
    in the paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    For detail see: https://arxiv.org/pdf/1801.01290.pdf

    This implementation slightly differs from the implementation of Soft Actor-Critic algorithm presented
    also by T. Haarnoja, et al. in the following paper: https://arxiv.org/pdf/1812.05905.pdf

    You will need to scale the reward received from the environment properly to get the algorithm work.
    '''

    def __init__(self, env_or_env_info,
                 v_function_builder=default_v_function_builder,
                 q_function_builder=default_q_function_builder,
                 policy_builder=default_policy_builder,
                 params=ICML2018SACParam()):
        super(ICML2018SAC, self).__init__(env_or_env_info, params=params)

        def v_function_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._v = v_function_builder(scope_name="v", env_info=self._env_info, algorithm_params=self._params)
        self._v_solver = {self._v.scope_name: v_function_solver_builder()}
        self._target_v = self._v.deepcopy('target_' + self._v.scope_name)
        assert isinstance(self._v, VFunction)
        assert isinstance(self._target_v, VFunction)

        def q_function_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._q1 = q_function_builder(scope_name="q1", env_info=self._env_info, algorithm_params=self._params)
        self._q2 = q_function_builder(scope_name="q2", env_info=self._env_info, algorithm_params=self._params)
        assert isinstance(self._q1, QFunction)
        assert isinstance(self._q2, QFunction)
        self._train_q_functions = [self._q1, self._q2]
        self._train_q_solvers = {}
        for q in self._train_q_functions:
            self._train_q_solvers[q.scope_name] = q_function_solver_builder()

        def policy_solver_builder():
            return NS.Adam(alpha=self._params.learning_rate)
        self._pi = policy_builder(scope_name="pi", env_info=self._env_info, algorithm_params=self._params)
        self._pi_solver = {self._pi.scope_name: policy_solver_builder()}
        assert isinstance(self._pi, StochasticPolicy)

        self._replay_buffer = ReplayBuffer(capacity=params.replay_buffer_size)

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None

        explorer_params = EE.RawPolicyExplorerParam(
            warmup_random_steps=self._params.start_timesteps,
            reward_scalar=self._params.reward_scalar,
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_greedy_action,
                                        env_info=self._env_info,
                                        params=explorer_params)
        return explorer

    def _setup_policy_training(self, env_or_buffer):
        # NOTE: Fix temperature to 1.0. Because This version of SAC adjusts it by scaling the reward
        policy_trainer_params = MT.policy_trainers.SoftPolicyTrainerParam(fixed_temperature=True)
        temperature = MT.policy_trainers.soft_policy_trainer.AdjustableTemperature(
            scope_name='temperature',
            initial_value=1.0)
        policy_trainer = MT.policy_trainers.SoftPolicyTrainer(env_info=self._env_info,
                                                              params=policy_trainer_params,
                                                              temperature=temperature,
                                                              q_functions=self._train_q_functions)

        training = MT.model_trainer.Training()
        policy_trainer.setup_training(self._pi, self._pi_solver, training)
        return policy_trainer

    def _setup_q_function_training(self, env_or_buffer):
        q_function_trainer_param = MT.q_value_trainers.SquaredTDQFunctionTrainerParam(
            reduction_method='mean',
            q_loss_scalar=0.5)
        q_function_trainer = MT.q_value_trainers.SquaredTDQFunctionTrainer(
            self._env_info,
            params=q_function_trainer_param)

        training = MT.q_value_trainings.VFunctionTargetedTraining(
            train_functions=self._train_q_functions,
            target_functions=self._target_v)
        # Update target_v in conjunction with q training
        training = MT.common_extensions.PeriodicalTargetUpdate(
            training,
            src_models=self._v,
            dst_models=self._target_v,
            target_update_frequency=self._params.target_update_interval,
            tau=self._params.tau
        )
        q_function_trainer.setup_training(self._train_q_functions, self._train_q_solvers, training)
        return q_function_trainer

    def _setup_v_function_training(self, env_or_buffer):
        v_function_trainer_params = MT.v_value_trainers.SquaredTDVFunctionTrainerParam(
            reduction_method='mean',
            v_loss_scalar=0.5
        )
        v_function_trainer = MT.v_value_trainers.SquaredTDVFunctionTrainer(
            env_info=self._env_info,
            params=v_function_trainer_params)

        training = MT.v_value_trainings.SoftVTraining(
            train_functions=self._v,
            target_functions=self._train_q_functions,  # Set training q as target
            target_policy=self._pi)
        v_function_trainer.setup_training(self._v, self._v_solver, training)
        copy_network_parameters(self._v.get_parameters(), self._target_v.get_parameters(), 1.0)

        return v_function_trainer

    @eval_api
    def compute_eval_action(self, state):
        action, _ = self._compute_greedy_action(state, deterministic=True)
        return action

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

        # Train in the order of v -> q -> policy
        self._v_function_trainer.train(batch)
        errors = self._q_function_trainer.train(batch)
        self._policy_trainer.train(batch)

        td_error = np.abs(errors['td_error'])
        replay_buffer.update_priorities(td_error)

    def _compute_greedy_action(self, s, deterministic=False):
        # evaluation input/action variables
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
        models = [self._v, self._target_v, self._q1, self._q2, self._pi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._train_q_solvers)
        solvers.update(self._v_solver)
        solvers.update(self._pi_solver)
        return solvers
