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
from typing import Any, Dict, List, Union

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.functions as RF
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedEnvironmentException
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import (BCQPerturbator, BCQVariationalAutoEncoder, DeterministicPolicy, Perturbator, QFunction,
                              TD3QFunction, VariationalAutoEncoder)
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.misc import sync_model


@dataclass
class BCQConfig(AlgorithmConfig):
    '''BCQConfig
    List of configurations for BCQ algorithm

    Args:
        gamma (float): discount factor of reward. Defaults to 0.99.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size (int): training batch size. Defaults to 100.
        tau (float): target network's parameter update coefficient. Defaults to 0.005.
        lmb (float): weight :math:`\\lambda` used for balancing the ratio between :math:`\\min{Q}` and :math:`\\max{Q}`\
            on target q value generation (i.e. :math:`\\lambda\\min{Q} + (1 - \\lambda)\\max{Q}`).\
            Defaults to 0.75.
        phi (float): action perturbator noise coefficient. Defaults to 0.05.
        num_q_ensembles (int): number of q function ensembles . Defaults to 2.
        num_action_samples (int): number of actions to sample for computing target q values. Defaults to 10.
    '''
    gamma: float = 0.99
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    tau: float = 0.005
    lmb: float = 0.75
    phi: float = 0.05
    num_q_ensembles: int = 2
    num_action_samples: int = 10

    def __post_init__(self):
        '''__post_init__

        Check set values are in valid range.

        '''
        self._assert_between(self.tau, 0.0, 1.0, 'tau')
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.lmb, 'lmb')
        self._assert_positive(self.phi, 'phi')
        self._assert_positive(self.num_q_ensembles, 'num_q_ensembles')
        self._assert_positive(self.num_action_samples, 'num_action_samples')
        self._assert_positive(self.batch_size, 'batch_size')


class DefaultQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BCQConfig,
                    **kwargs) -> QFunction:
        return TD3QFunction(scope_name)


class DefaultVAEBuilder(ModelBuilder[VariationalAutoEncoder]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BCQConfig,
                    **kwargs) -> VariationalAutoEncoder:
        max_action_value = float(env_info.action_space.high[0])
        return BCQVariationalAutoEncoder(scope_name,
                                         env_info.state_dim,
                                         env_info.action_dim,
                                         env_info.action_dim*2,
                                         max_action_value)


class DefaultPerturbatorBuilder(ModelBuilder[Perturbator]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: BCQConfig,
                    **kwargs) -> Perturbator:
        max_action_value = float(env_info.action_space.high[0])
        return BCQPerturbator(scope_name,
                              env_info.state_dim,
                              env_info.action_dim,
                              max_action_value)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: BCQConfig,
                     **kwargs):
        return NS.Adam(alpha=algorithm_config.learning_rate)


class BCQ(Algorithm):
    '''Batch-Constrained Q-learning (BCQ) algorithm

    This class implements the Batch-Constrained Q-learning (BCQ) algorithm
    proposed by S. Fujimoto, et al. in the paper: "Off-Policy Deep Reinforcement Learning without Exploration"
    For details see: https://arxiv.org/abs/1812.02900

    This algorithm only supports offline training.

    Args:
        env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`BCQConfig <nnabla_rl.algorithms.bcq.BCQConfig>`):
            configuration of the BCQ algorithm
        q_function_builder (:py:class:`ModelBuilder[QFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of q-function models
        q_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for q-function solvers
        vae_builder (:py:class:`ModelBuilder[VariationalAutoEncoder] <nnabla_rl.builders.ModelBuilder>`):
            builder of variational auto encoder models
        vae_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for variational auto encoder solvers
        perturbator_builder (:py:class:`PerturbatorBuilder <nnabla_rl.builders.PerturbatorBuilder>`):
            builder of perturbator models
        perturbator_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder for perturbator solvers
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: BCQConfig
    _q_ensembles: List[QFunction]
    _q_solvers: Dict[str, nn.solver.Solver]
    _target_q_ensembles: List[QFunction]
    _vae: VariationalAutoEncoder
    _vae_solver: nn.solver.Solver
    _xi: Perturbator
    _xi_solver: nn.solver.Solver

    _q_function_trainer: ModelTrainer
    _encoder_trainer: ModelTrainer
    _perturbator_trainer: ModelTrainer

    _eval_state_var: nn.Variable
    _eval_action: nn.Variable
    _eval_max_index: nn.Variable

    _encoder_trainer_state: Dict[str, Any]
    _q_function_trainer_state: Dict[str, Any]
    _perturbator_trainer_state: Dict[str, Any]

    def __init__(self,
                 env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: BCQConfig = BCQConfig(),
                 q_function_builder: ModelBuilder[QFunction] = DefaultQFunctionBuilder(),
                 q_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 vae_builder: ModelBuilder[VariationalAutoEncoder] = DefaultVAEBuilder(),
                 vae_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 perturbator_builder: ModelBuilder[Perturbator] = DefaultPerturbatorBuilder(),
                 perturbator_solver_builder: SolverBuilder = DefaultSolverBuilder()):
        super(BCQ, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_discrete_action_env():
            raise UnsupportedEnvironmentException

        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._q_ensembles = []
            self._q_solvers = {}
            self._target_q_ensembles = []

            for i in range(self._config.num_q_ensembles):
                q = q_function_builder(scope_name=f"q{i}",
                                       env_info=self._env_info,
                                       algorithm_config=self._config)
                target_q = q.deepcopy(f'target_q{i}')
                assert isinstance(target_q, QFunction)
                self._q_ensembles.append(q)
                self._q_solvers[q.scope_name] = q_solver_builder(env_info=self._env_info, algorithm_config=self._config)
                self._target_q_ensembles.append(target_q)

            self._vae = vae_builder(scope_name="vae", env_info=self._env_info, algorithm_config=self._config)
            self._vae_solver = vae_solver_builder(env_info=self._env_info, algorithm_config=self._config)

            self._xi = perturbator_builder(scope_name="xi", env_info=self._env_info, algorithm_config=self._config)
            self._xi_solver = perturbator_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._target_xi = perturbator_builder(
                scope_name="target_xi", env_info=self._env_info, algorithm_config=self._config)

    @eval_api
    def compute_eval_action(self, s):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            s = np.expand_dims(s, axis=0)
            if not hasattr(self, '_eval_state_var'):
                self._eval_state_var = nn.Variable(s.shape)
                repeat_num = 100
                state = RF.repeat(x=self._eval_state_var, repeats=repeat_num, axis=0)
                assert state.shape == (repeat_num, self._eval_state_var.shape[1])
                actions = self._vae.decode(z=None, state=state)
                noise = self._xi.generate_noise(state, actions, self._config.phi)
                self._eval_action = actions + noise
                q_values = self._q_ensembles[0].q(state, self._eval_action)
                self._eval_max_index = RF.argmax(q_values, axis=0)
            self._eval_state_var.d = s
            nn.forward_all([self._eval_action, self._eval_max_index])
            return self._eval_action.d[self._eval_max_index.d[0]]

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._encoder_trainer = self._setup_encoder_training(env_or_buffer)
        self._q_function_trainer = self._setup_q_function_training(env_or_buffer)
        self._perturbator_trainer = self._setup_perturbator_training(env_or_buffer)

    def _setup_encoder_training(self, env_or_buffer):
        trainer_config = MT.encoder_trainers.KLDVariationalAutoEncoderTrainerConfig()

        encoder_trainer = MT.encoder_trainers.KLDVariationalAutoEncoderTrainer(
            models=self._vae,
            solvers={self._vae.scope_name: self._vae_solver},
            env_info=self._env_info,
            config=trainer_config)
        return encoder_trainer

    def _setup_q_function_training(self, env_or_buffer):
        trainer_config = MT.q_value.BCQQTrainerConfig(reduction_method='mean',
                                                      num_action_samples=self._config.num_action_samples,
                                                      lmb=self._config.lmb)

        # This is a wrapper class which outputs the target action for next state in q function training
        class PerturbedPolicy(DeterministicPolicy):
            def __init__(self, vae, perturbator, phi):
                self._vae = vae
                self._perturbator = perturbator
                self._phi = phi

            def pi(self, s):
                a = self._vae.decode(z=None, state=s)
                noise = self._perturbator.generate_noise(s, a, phi=self._phi)
                return a + noise
        target_policy = PerturbedPolicy(self._vae, self._target_xi, self._config.phi)
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

    def _setup_perturbator_training(self, env_or_buffer):
        trainer_config = MT.perturbator_trainers.BCQPerturbatorTrainerConfig(
            phi=self._config.phi
        )

        perturbator_trainer = MT.perturbator.BCQPerturbatorTrainer(
            models=self._xi,
            solvers={self._xi.scope_name: self._xi_solver},
            q_function=self._q_ensembles[0],
            vae=self._vae,
            env_info=self._env_info,
            config=trainer_config)
        sync_model(self._xi, self._target_xi, 1.0)
        return perturbator_trainer

    def _run_online_training_iteration(self, env):
        raise NotImplementedError('BCQ does not support online training')

    def _run_offline_training_iteration(self, buffer):
        self._bcq_training(buffer)

    def _bcq_training(self, replay_buffer):
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

        # Train vae
        self._encoder_trainer_state = self._encoder_trainer.train(batch)

        self._q_function_trainer_state = self._q_function_trainer.train(batch)
        for q, target_q in zip(self._q_ensembles, self._target_q_ensembles):
            sync_model(q, target_q, tau=self._config.tau)
        td_errors = np.abs(self._q_function_trainer_state['td_errors'])
        replay_buffer.update_priorities(td_errors)

        self._perturbator_trainer.train(batch)
        sync_model(self._xi, self._target_xi, tau=self._config.tau)

        self._perturbator_trainer_state = self._perturbator_trainer.train(batch)

    def _models(self):
        models = [*self._q_ensembles, *self._target_q_ensembles,
                  self._vae, self._xi, self._target_xi]
        return {model.scope_name: model for model in models}

    def _solvers(self):
        solvers = {}
        solvers.update(self._q_solvers)
        solvers[self._vae.scope_name] = self._vae_solver
        solvers[self._xi.scope_name] = self._xi_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(BCQ, self).latest_iteration_state
        if hasattr(self, '_encoder_trainer_state'):
            latest_iteration_state['scalar'].update({'encoder_loss': self._encoder_trainer_state['encoder_loss']})
        if hasattr(self, '_perturbator_trainer_state'):
            latest_iteration_state['scalar'].update(
                {'perturbator_loss': self._perturbator_trainer_state['perturbator_loss']})
        if hasattr(self, '_q_function_trainer_state'):
            latest_iteration_state['scalar'].update({'q_loss': self._q_function_trainer_state['q_loss']})
            latest_iteration_state['histogram'].update(
                {'td_errors': self._q_function_trainer_state['td_errors'].flatten()})
        return latest_iteration_state


if __name__ == "__main__":
    import nnabla_rl.environments as E
    env = E.DummyContinuous()
    bcq = BCQ(env)
