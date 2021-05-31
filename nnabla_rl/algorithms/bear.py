# Copyright 2020,2021 Sony Corporation.
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
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.solvers as NS
import nnabla_rl.functions as RF
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedEnvironmentException
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import (BEARPolicy, DeterministicPolicy, QFunction, StochasticPolicy, TD3QFunction,
                              UnsquashedVariationalAutoEncoder, VariationalAutoEncoder)
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class BEARConfig(AlgorithmConfig):
    '''BEARConfig
    List of configurations for BEAR algorithm.

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size (int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        lmb (float): weight :math:`\\lambda` used for balancing the ratio between :math:`\\min{Q}` and :math:`\\max{Q}`\
            on target q value generation (i.e. :math:`\\lambda\\min{Q} + (1 - \\lambda)\\max{Q}`).\
            Defaults to 0.75.
        epsilon (float): inequality constraint of dual gradient descent. Defaults to 0.05.
        num_q_ensembles (int): number of q ensembles . Defaults to 2.
        num_mmd_actions (int): number of actions to sample for computing maximum mean discrepancy (MMD). Defaults to 5.
        num_action_samples (int): number of actions to sample for computing target q values. Defaults to 10.
        mmd_type (str): kernel type used for MMD computation. laplacian or gaussian is supported. Defaults to gaussian.
        mmd_sigma (float): parameter used for adjusting the  MMD. Defaults to 20.0.
        initial_lagrange_multiplier (float, optional): Initial value of lagrange multiplier. \
            If not specified, random value sampled from normal distribution will be used instead.
        fix_lagrange_multiplier (bool): Either to fix the lagrange multiplier or not. Defaults to False.
        warmup_iterations (int): Number of iterations until start updating the policy. Defaults to 20000
        use_mean_for_eval (bool): Use mean value instead of best action among the samples for evaluation.\
            Defaults to False.
    '''
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 100
    tau: float = 0.005
    lmb: float = 0.75
    epsilon: float = 0.05
    num_q_ensembles: int = 2
    num_mmd_actions: int = 5
    num_action_samples: int = 10
    mmd_type: str = 'gaussian'
    mmd_sigma: float = 20.0
    initial_lagrange_multiplier: Optional[float] = None
    fix_lagrange_multiplier: bool = False
    warmup_iterations: int = 20000
    use_mean_for_eval: bool = False

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        if not ((0.0 <= self.tau) & (self.tau <= 1.0)):
            raise ValueError('tau must lie between [0.0, 1.0]')
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        if not (0 <= self.num_q_ensembles):
            raise ValueError('num q ensembles must not be negative')
        if not (0 <= self.num_mmd_actions):
            raise ValueError('num mmd actions must not be negative')
        if not (0 <= self.num_action_samples):
            raise ValueError('num action samples must not be negative')
        if not (0 <= self.warmup_iterations):
            raise ValueError('warmup iterations must not be negative')
        if not (0 <= self.batch_size):
            raise ValueError('batch size must not be negative')


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BEARConfig,
                    **kwargs) -> QFunction:
        return TD3QFunction(scope_name)


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BEARConfig,
                    **kwargs) -> StochasticPolicy:
        return BEARPolicy(scope_name, env_info.action_dim)


class DefaultVAEBuilder(ModelBuilder[VariationalAutoEncoder]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BEARConfig,
                    **kwargs) -> VariationalAutoEncoder:
        return UnsquashedVariationalAutoEncoder(scope_name,
                                                env_info.state_dim,
                                                env_info.action_dim,
                                                env_info.action_dim*2)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: BEARConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


class BEAR(Algorithm):
    '''Bootstrapping Error Accumulation Reduction (BEAR) algorithm.

    This class implements the Bootstrapping Error Accumulation Reduction (BEAR) algorithm
    proposed by A. Kumar, et al. in the paper: "Stabilizing Off-Policy Q-learning via Bootstrapping Error Reduction"
    For details see: https://arxiv.org/abs/1906.00949

    This algorithm only supports offline training.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`BEARConfig <nnabla_rl.algorithms.bear.BEARConfig>`):
            configuration of the BEAR algorithm
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q-function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for q-function solvers
        pi_function_builder (:py:class:`ModelBuilder[StochasticPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
        pi_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for policy solvers
        vae_builder (:py:class:`ModelBuilder[VariationalAutoEncoder] <nnabla_rl.builders.ModelBuilder>`):
            builder of variational auto encoder models
        vae_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for variational auto encoder solvers
        lagrange_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for lagrange multiplier solver
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: BEARConfig
    _q_ensembles: List[QFunction]
    _q_solvers: Dict[str, nn.solver.Solver]
    _target_q_ensembles: List[QFunction]
    _pi: StochasticPolicy
    _pi_solver: nn.solver.Solver
    _target_pi: StochasticPolicy
    _vae: VariationalAutoEncoder
    _vae_solver: nn.solver.Solver
    _lagrange: MT.policy_trainers.bear_policy_trainer.AdjustableLagrangeMultiplier
    _lagrange_solver: nn.solver.Solver
    _q_function_trainer: ModelTrainer
    _encoder_trainer: ModelTrainer
    _policy_trainer: ModelTrainer
    _eval_state_var: nn.Variable
    _eval_action: nn.Variable
    _eval_max_index: nn.Variable

    _encoder_trainer_state: Dict[str, Any]
    _policy_trainer_state: Dict[str, Any]
    _q_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: BEARConfig = BEARConfig(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 pi_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 pi_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 vae_builder: ModelBuilder[VariationalAutoEncoder] = DefaultVAEBuilder(),
                 vae_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 lagrange_solver_builder: SolverBuilder = DefaultSolverBuilder()):
        super(BEAR, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_discrete_action_env():
            raise UnsupportedEnvironmentException

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._q_ensembles = []
            self._q_solvers = {}
            self._target_q_ensembles = []
            for i in range(self._config.num_q_ensembles):
                q = q_function_builder(scope_name="q{}".format(
                    i), env_info=self._env_info, algorithm_config=self._config)
                target_q = q_function_builder(
                    scope_name="target_q{}".format(i), env_info=self._env_info, algorithm_config=self._config)
                self._q_ensembles.append(q)
                self._q_solvers[q.scope_name] = q_solver_builder(env_info=self._env_info, algorithm_config=self._config)
                self._target_q_ensembles.append(target_q)

            self._pi = pi_builder(scope_name="pi", env_info=self._env_info, algorithm_config=self._config)
            self._pi_solver = pi_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._target_pi = pi_builder(scope_name="target_pi", env_info=self._env_info, algorithm_config=self._config)

            self._vae = vae_builder(scope_name="vae", env_info=self._env_info, algorithm_config=self._config)
            self._vae_solver = vae_solver_builder(env_info=self._env_info, algorithm_config=self._config)

            self._lagrange = MT.policy_trainers.bear_policy_trainer.AdjustableLagrangeMultiplier(
                scope_name="alpha",
                initial_value=self._config.initial_lagrange_multiplier)
            self._lagrange_solver = lagrange_solver_builder(env_info=self._env_info, algorithm_config=self._config)

    @eval_api
    def compute_eval_action(self, s):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            s = np.expand_dims(s, axis=0)
            if not hasattr(self, '_eval_state_var'):
                self._eval_state_var = nn.Variable(s.shape)
                if self._config.use_mean_for_eval:
                    eval_distribution = self._pi.pi(self._eval_state_var)
                    self._eval_action = NF.tanh(eval_distribution.mean())
                else:
                    repeat_num = 100
                    state = RF.repeat(x=self._eval_state_var, repeats=repeat_num, axis=0)
                    assert state.shape == (repeat_num, self._eval_state_var.shape[1])
                    eval_distribution = self._pi.pi(state)
                    self._eval_action = NF.tanh(eval_distribution.sample())
                    q_values = self._q_ensembles[0].q(state, self._eval_action)
                    self._eval_max_index = RF.argmax(q_values, axis=0)
            self._eval_state_var.d = s
            if self._config.use_mean_for_eval:
                self._eval_action.forward()
                return np.squeeze(self._eval_action.d, axis=0)
            else:
                nn.forward_all([self._eval_action, self._eval_max_index])
                return self._eval_action.d[self._eval_max_index.d[0]]

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._encoder_trainer = self._setup_encoder_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)

    def _setup_encoder_training(self, env_or_buffer):
        trainer_config = MT.encoder_trainers.KLDVariationalAutoEncoderTrainerConfig()

        # Wrapper for squashing reconstructed action during vae training
        class SquashedActionVAE(VariationalAutoEncoder):
            def __init__(self, original_vae):
                super().__init__(original_vae.scope_name)
                self._original_vae = original_vae

            def encode_and_decode(self, s, **kwargs):
                latent_distribution, reconstructed = self._original_vae.encode_and_decode(s, **kwargs)
                return latent_distribution, NF.tanh(reconstructed)

            def encode(self, *args): raise NotImplementedError
            def decode(self, *args): raise NotImplementedError
            def decode_multiple(self, decode_num, *args): raise NotImplementedError
            def latent_distribution(self, *args): raise NotImplementedError

        squashed_action_vae = SquashedActionVAE(self._vae)
        encoder_trainer = MT.encoder_trainers.KLDVariationalAutoEncoderTrainer(
            models=squashed_action_vae,
            solvers={self._vae.scope_name: self._vae_solver},
            env_info=self._env_info,
            config=trainer_config)
        return encoder_trainer

    def _setup_q_function_training(self, env_or_buffer):
        # This is a wrapper class which outputs the target action for next state in q function training
        class PerturbedPolicy(DeterministicPolicy):
            def __init__(self, target_pi):
                super().__init__(target_pi.scope_name)
                self._target_pi = target_pi

            def pi(self, s):
                policy_distribution = self._target_pi.pi(s)
                return NF.tanh(policy_distribution.sample())
        target_policy = PerturbedPolicy(self._target_pi)

        trainer_config = MT.q_value.BCQQTrainerConfig(reduction_method='mean',
                                                      num_action_samples=self._config.num_action_samples,
                                                      lmb=self._config.lmb)
        q_function_trainer = MT.q_value.BCQQTrainer(
            train_functions=self._q_ensembles,
            solvers=self._q_solvers,
            target_functions=self._target_q_ensembles,
            target_policy=target_policy,
            env_info=self._env_info,
            config=trainer_config)
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            sync_model(q, target_q, 1.0)
        return q_function_trainer

    def _setup_policy_training(self, env_or_buffer):
        trainer_config = MT.policy_trainers.BEARPolicyTrainerConfig(
            num_mmd_actions=self._config.num_mmd_actions,
            mmd_type=self._config.mmd_type,
            epsilon=self._config.epsilon,
            fix_lagrange_multiplier=self._config.fix_lagrange_multiplier,
            warmup_iterations=self._config.warmup_iterations-self._iteration_num)

        class SquashedActionQ(QFunction):
            def __init__(self, original_q):
                super().__init__(original_q.scope_name)
                self._original_q = original_q

            def q(self, s, a):
                squashed_action = NF.tanh(a)
                return self._original_q.q(s, squashed_action)

        wrapped_qs = [SquashedActionQ(q) for q in self._q_ensembles]
        policy_trainer = MT.policy_trainers.BEARPolicyTrainer(
            models=self._pi,
            solvers={self._pi.scope_name: self._pi_solver},
            q_ensembles=wrapped_qs,
            vae=self._vae,
            lagrange_multiplier=self._lagrange,
            lagrange_solver=self._lagrange_solver,
            env_info=self._env_info,
            config=trainer_config)
        sync_model(self._pi, self._target_pi, 1.0)

        return policy_trainer

    def _run_online_training_iteration(self, env):
        raise NotImplementedError

    def _run_offline_training_iteration(self, buffer):
        self._bear_training(buffer)

    def _bear_training(self, replay_buffer):
        experiences, info = replay_buffer.sample(self._config.batch_size)
        (s, a, r, non_terminal, s_next, *_) = marshal_experiences(experiences)
        batch = TrainingBatch(batch_size=self._config.batch_size,
                              s_current=s,
                              a_current=a,
                              gamma=self._config.gamma,
                              reward=r,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=info['weights'])

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            sync_model(q, target_q, tau=self._config.tau)
        td_errors = np.abs(self._q_function_trainer_state['td_errors'])
        replay_buffer.update_priorities(td_errors)

        self._encoder_trainer_state = self._encoder_trainer.train(batch)
        self._policy_trainer_state = self._policy_trainer.train(batch)
        sync_model(self._pi, self._target_pi, tau=self._config.tau)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._pi, self._target_pi, self._vae,
                  self._lagrange]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._q_solvers)
        solvers[self._pi.scope_name] = self._pi_solver
        solvers[self._vae.scope_name] = self._vae_solver
        if not self._config.fix_lagrange_multiplier:
            solvers[self._lagrange.scope_name] = self._lagrange_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(BEAR, self).latest_iteration_state
        if hasattr(self, '_encoder_trainer_state'):
            latest_iteration_state['scalar'].update({'encoder_loss': self._encoder_trainer_state['encoder_loss']})
        if hasattr(self, '_policy_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_loss': self._policy_trainer_state['pi_loss']})
        if hasattr(self, '_q_function_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': self._q_function_trainer_state['q_loss']})
            latest_iteration_state['histogram'].update({'td_errors': self._q_function_trainer_state['td_errors']})
        return latest_iteration_state
