import nnabla as nn

import nnabla.solvers as NS

import numpy as np

from dataclasses import dataclass

import gym
from typing import Optional, Union

from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.builders import StochasticPolicyBuilder, VFunctionBuilder, SolverBuilder, PreprocessorBuilder
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.utils.data import marshall_experiences
from nnabla_rl.algorithms.common_utils import \
    compute_v_target_and_advantage, _StatePreprocessedPolicy, _StatePreprocessedVFunction
from nnabla_rl.models import TRPOPolicy, TRPOVFunction, StochasticPolicy, VFunction, Model
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
import nnabla_rl.environment_explorers as EE
import nnabla_rl.model_trainers as MT
import nnabla_rl.preprocessors as RP


@dataclass
class TRPOParam(AlgorithmParam):
    gpu_batch_size: Optional[int] = None
    gamma: float = 0.995
    lmb: float = 0.97
    pi_batch_size: int = 5000
    num_steps_per_iteration: int = 5000
    sigma_kl_divergence_constraint: float = 0.01
    maximum_backtrack_numbers: int = 10
    conjugate_gradient_damping: float = 0.1
    conjugate_gradient_iterations: int = 20
    vf_epochs: int = 5
    vf_batch_size: int = 64
    vf_learning_rate: float = 1e-3
    preprocess_state: bool = True

    def __post_init__(self):
        '''__post_init__

        Check the values are in valid range.

        '''
        self._assert_between(self.pi_batch_size, 0, self.num_steps_per_iteration, 'pi_batch_size')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.lmb, 0.0, 1.0, 'lmb')
        self._assert_positive(self.num_steps_per_iteration, 'num_steps_per_iteration')
        self._assert_between(self.pi_batch_size, 0, self.num_steps_per_iteration, 'pi_batch_size')
        self._assert_positive(self.sigma_kl_divergence_constraint, 'sigma_kl_divergence_constraint')
        self._assert_positive(self.maximum_backtrack_numbers, 'maximum_backtrack_numbers')
        self._assert_positive(self.conjugate_gradient_damping, 'conjugate_gradient_damping')
        self._assert_positive(self.conjugate_gradient_iterations, 'conjugate_gradient_iterations')
        self._assert_positive(self.vf_epochs, 'vf_epochs')
        self._assert_positive(self.vf_batch_size, 'vf_batch_size')
        self._assert_positive(self.vf_learning_rate, 'vf_learning_rate')


class DefaultPolicyBuilder(StochasticPolicyBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: TRPOParam,
                    **kwargs) -> StochasticPolicy:
        return TRPOPolicy(scope_name, env_info.action_dim)


class DefaultVFunctionBuilder(VFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: TRPOParam,
                    **kwargs) -> VFunction:
        return TRPOVFunction(scope_name)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_params: TRPOParam,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_params.vf_learning_rate)


class DefaultPreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(self,
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_params: AlgorithmParam,
                           **kwargs) -> Preprocessor:
        return RP.RunningMeanNormalizer(scope_name, env_info.state_shape, value_clip=(-5.0, 5.0))


class TRPO(Algorithm):
    """ Trust Region Policy Optimiation method
        with Generalized Advantage Estimation (GAE)
        See: https://arxiv.org/pdf/1502.05477.pdf and https://arxiv.org/pdf/1506.02438.pdf
    """

    _params: TRPOParam
    _policy: StochasticPolicy
    _v_function: VFunction
    _v_function_solver: nn.solvers.Solver
    _state_preprocessor: Optional[Preprocessor]
    _environment_explorer: EnvironmentExplorer
    _policy_trainer: ModelTrainer
    _v_function_trainer: ModelTrainer
    _eval_state_var: nn.Variable
    _eval_action: nn.Variable

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 params: TRPOParam = TRPOParam(),
                 v_function_builder: VFunctionBuilder = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: StochasticPolicyBuilder = DefaultPolicyBuilder(),
                 state_preprocessor_builder: Optional[PreprocessorBuilder] = DefaultPreprocessorBuilder()):
        super(TRPO, self).__init__(env_or_env_info, params=params)
        if self._env_info.is_discrete_action_env():
            raise NotImplementedError

        self._v_function = v_function_builder('v', self._env_info, self._params)
        self._policy = policy_builder('pi', self._env_info, self._params)

        self._preprocessor: Optional[Preprocessor] = None
        if self._params.preprocess_state and state_preprocessor_builder is not None:
            preprocessor = state_preprocessor_builder('preprocessor', self._env_info, self._params)
            assert preprocessor is not None
            self._v_function = _StatePreprocessedVFunction(v_function=self._v_function, preprocessor=preprocessor)
            self._policy = _StatePreprocessedPolicy(policy=self._policy, preprocessor=preprocessor)
            self._state_preprocessor = preprocessor
        self._v_function_solver = v_solver_builder(self._env_info, self._params)

    def _before_training_start(self, env_or_buffer):
        self._environment_explorer = self._setup_environment_explorer(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_environment_explorer(self, env_or_buffer):
        if self._is_buffer(env_or_buffer):
            return None
        explorer_params = EE.RawPolicyExplorerParam(
            initial_step_num=self.iteration_num,
            timelimit_as_terminal=False
        )
        explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                        env_info=self._env_info,
                                        params=explorer_params)
        return explorer

    def _setup_v_function_training(self, env_or_buffer):
        v_function_trainer_params = MT.v_value_trainers.SquaredTDVFunctionTrainerParam(
            reduction_method='mean',
            v_loss_scalar=1.0
        )
        v_function_trainer = MT.v_value_trainers.SquaredTDVFunctionTrainer(
            env_info=self._env_info,
            params=v_function_trainer_params)

        training = MT.v_value_trainings.MonteCarloVValueTraining()
        v_function_trainer.setup_training(
            self._v_function, {self._v_function.scope_name: self._v_function_solver}, training)
        return v_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.TRPOPolicyTrainerParam(
            gpu_batch_size=self._params.gpu_batch_size,
            sigma_kl_divergence_constraint=self._params.sigma_kl_divergence_constraint,
            maximum_backtrack_numbers=self._params.maximum_backtrack_numbers,
            conjugate_gradient_damping=self._params.conjugate_gradient_damping,
            conjugate_gradient_iterations=self._params.conjugate_gradient_iterations)
        policy_trainer = MT.policy_trainers.TRPOPolicyTrainer(env_info=self._env_info,
                                                              params=policy_trainer_params)
        training = MT.model_trainer.Training()
        policy_trainer.setup_training(self._policy, {}, training)

        return policy_trainer

    @eval_api
    def compute_eval_action(self, s):
        action, _ = self._compute_action(s)
        return action

    def _run_online_training_iteration(self, env):
        if self.iteration_num % self._params.num_steps_per_iteration != 0:
            return

        buffer = ReplayBuffer(capacity=self._params.num_steps_per_iteration)

        num_steps = 0
        while num_steps <= self._params.num_steps_per_iteration:
            experience = self._environment_explorer.rollout(env)
            buffer.append(experience)
            num_steps += len(experience)

        self._trpo_training(buffer)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _trpo_training(self, buffer):
        buffer_iterator = BufferIterator(buffer, 1, shuffle=False, repeat=False)
        s, a, v_target, advantage = self._align_experiences(buffer_iterator)

        if self._params.preprocess_state:
            self._state_preprocessor.update(s)

        # v function training
        self._v_function_training(s, v_target)

        # policy training
        self._policy_training(s, a, v_target, advantage)

    def _align_experiences(self, buffer_iterator):
        v_target_batch, adv_batch = self._compute_v_target_and_advantage(buffer_iterator)

        s_batch, a_batch = self._align_state_and_action(buffer_iterator)

        return s_batch[:self._params.num_steps_per_iteration], \
            a_batch[:self._params.num_steps_per_iteration], \
            v_target_batch[:self._params.num_steps_per_iteration], \
            adv_batch[:self._params.num_steps_per_iteration]

    def _compute_v_target_and_advantage(self, buffer_iterator):
        v_target_batch = []
        adv_batch = []
        buffer_iterator.reset()
        for experiences, _ in buffer_iterator:
            # length of experiences is 1
            v_target, adv = compute_v_target_and_advantage(
                self._v_function, experiences[0], gamma=self._params.gamma, lmb=self._params.lmb)
            v_target_batch.append(v_target.reshape(-1, 1))
            adv_batch.append(adv.reshape(-1, 1))

        adv_batch = np.concatenate(adv_batch, axis=0)
        v_target_batch = np.concatenate(v_target_batch, axis=0)

        adv_mean = np.mean(adv_batch)
        adv_std = np.std(adv_batch)
        adv_batch = (adv_batch - adv_mean) / adv_std
        return v_target_batch, adv_batch

    def _align_state_and_action(self, buffer_iterator):
        s_batch = []
        a_batch = []

        buffer_iterator.reset()
        for experiences, _ in buffer_iterator:
            # length of experiences is 1
            s_seq, a_seq, *_ = marshall_experiences(experiences[0])
            s_batch.append(s_seq)
            a_batch.append(a_seq)

        s_batch = np.concatenate(s_batch, axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        return s_batch, a_batch

    def _v_function_training(self, s, v_target):
        num_iterations_per_epoch = self._params.num_steps_per_iteration // self._params.vf_batch_size

        for _ in range(self._params.vf_epochs * num_iterations_per_epoch):
            indices = np.random.randint(0, self._params.num_steps_per_iteration, size=self._params.vf_batch_size)
            batch = TrainingBatch(batch_size=self._params.vf_batch_size,
                                  s_current=s[indices],
                                  extra={'v_target': v_target[indices]})
            self._v_function_trainer.train(batch)

    def _policy_training(self, s, a, v_target, advantage):
        extra = {}
        extra['v_target'] = v_target[:self._params.pi_batch_size]
        extra['advantage'] = advantage[:self._params.pi_batch_size]
        batch = TrainingBatch(batch_size=self._params.pi_batch_size,
                              s_current=s[:self._params.pi_batch_size],
                              a_current=a[:self._params.pi_batch_size],
                              extra=extra)

        self._policy_trainer.train(batch)

    def _compute_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._policy.pi(self._eval_state_var)
            self._eval_action = distribution.sample()
        self._eval_state_var.d = s
        self._eval_action.forward()
        return np.squeeze(self._eval_action.d, axis=0), {}

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._v_function.scope_name] = self._v_function
        if self._params.preprocess_state and isinstance(self._state_preprocessor, Model):
            models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._v_function.scope_name] = self._v_function_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(TRPO, self).latest_iteration_state
        return latest_iteration_state
