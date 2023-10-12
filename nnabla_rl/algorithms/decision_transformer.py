# Copyright 2023 Sony Group Corporation.
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
from typing import Any, Dict, Optional, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.model_trainers as MT
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import _DecisionTransformerActionSelector
from nnabla_rl.builders import LearningRateSchedulerBuilder, ModelBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import AtariDecisionTransformer, DeterministicDecisionTransformer, StochasticDecisionTransformer
from nnabla_rl.replay_buffers import TrajectoryReplayBuffer
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences
from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm

DecisionTransformerModel = Union[StochasticDecisionTransformer, DeterministicDecisionTransformer]


@dataclass
class DecisionTransformerConfig(AlgorithmConfig):
    """List of configurations for DecisionTransformer algorithm.

    Args:
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0006.
        batch_size (int): training batch size. Defaults to 128.
        context_length (int): Context length of transformer model. Defaults to 30.
        max_timesteps (Optional[int]): Optional. Maximum timestep of training environmet.
            If the value is not provided, the algorithm will guess the maximum episode length through EnvironmentInfo.
        grad_clip_norm (float): Gradient clipping threshold for default solver. Defaults to 1.0.
        weight_decay (float): Weight decay parameter for default solver. Defaults to 0.1.
        target_return (int): Initial target return used to compute the evaluation action. Defaults to 90.
        reward_scale (float): Reward scaler. Reward received during evaluation will be multiplied by this value.
    """
    learning_rate: float = 6.0e-4
    batch_size: int = 128
    context_length: int = 30
    max_timesteps: Optional[int] = None
    grad_clip_norm: float = 1.0
    weight_decay: float = 0.1

    target_return: int = 90
    reward_scale: float = 1.0

    def __post_init__(self):
        """__post_init__

        Check set values are in valid range.
        """
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.context_length, 'context_length')
        self._assert_positive(self.grad_clip_norm, 'grad_clip_norm')
        if self.max_timesteps is not None:
            self._assert_positive(self.max_timesteps, 'max_timesteps')
        self._assert_positive(self.weight_decay, 'weight_decay')
        self._assert_positive(self.target_return, 'target_return')


class DefaultTransformerBuilder(ModelBuilder[DecisionTransformerModel]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: DecisionTransformerConfig,
                    **kwargs) -> DecisionTransformerModel:
        max_timesteps = cast(int, kwargs['max_timesteps'])
        return AtariDecisionTransformer(scope_name,
                                        env_info.action_dim,
                                        max_timestep=max_timesteps,
                                        context_length=algorithm_config.context_length)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: DecisionTransformerConfig,
                     **kwargs) -> nn.solver.Solver:
        solver = NS.Adam(alpha=algorithm_config.learning_rate, beta1=0.9, beta2=0.95)
        return AutoClipGradByNorm(solver, algorithm_config.grad_clip_norm)


class DecisionTransformer(Algorithm):
    """DecisionTransformer algorithm.

    This class implements the DecisionTransformer algorithm
    proposed by L. Chen, et al. in the paper: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
    For details see: https://arxiv.org/abs/2106.01345

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        config (:py:class:`DecisionTransformerConfig <nnabla_rl.algorithms.dqn.DecisionTransformerConfig>`):
            the parameter for DecisionTransformer training
        transformer_builder (:py:class:`ModelBuilder <nnabla_rl.builders.ModelBuilder>`): builder of transformer model
        solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder of transformer solver
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DecisionTransformerConfig
    _decision_transformer: DecisionTransformerModel
    _transformer_solver: nn.solver.Solver
    _decision_transformer_trainer: ModelTrainer
    _decision_transformer_trainer_state: Dict[str, Any]
    _action_selector: _DecisionTransformerActionSelector

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 config: DecisionTransformerConfig = DecisionTransformerConfig(),
                 transformer_builder: ModelBuilder[DecisionTransformerModel] = DefaultTransformerBuilder(),
                 transformer_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 transformer_wd_solver_builder: Optional[SolverBuilder] = None,
                 lr_scheduler_builder: Optional[LearningRateSchedulerBuilder] = None):
        super(DecisionTransformer, self).__init__(env_or_env_info, config=config)
        if config.max_timesteps is None:
            assert not np.isposinf(self._env_info.max_episode_steps)
            self._max_timesteps = self._env_info.max_episode_steps
        else:
            self._max_timesteps = config.max_timesteps
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._decision_transformer = transformer_builder(scope_name='decision_transformer',
                                                             env_info=self._env_info,
                                                             algorithm_config=self._config,
                                                             max_timesteps=self._max_timesteps)
            self._decision_transformer_solver = transformer_solver_builder(
                env_info=self._env_info, algorithm_config=self._config)
            self._decision_transformer_wd_solver = None if transformer_wd_solver_builder is None else \
                transformer_wd_solver_builder(env_info=self._env_info, algorithm_config=self._config)
        self._lr_scheduler = None if lr_scheduler_builder is None else lr_scheduler_builder(
            env_info=self._env_info, algorithm_config=self._config)

        self._action_selector = _DecisionTransformerActionSelector(self._env_info,
                                                                   self._decision_transformer.shallowcopy(),
                                                                   self._max_timesteps,
                                                                   self._config.context_length,
                                                                   self._config.target_return,
                                                                   self._config.reward_scale)

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}):
        if 'reward' not in extra_info:
            raise ValueError(f'{self.__name__} requires previous reward info in addition to state to compute action.'
                             'use extra_info["reward"]=reward')
        return self._action_selector(state, begin_of_episode=begin_of_episode, extra_info=extra_info)

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._decision_transformer_trainer = self._setup_decision_transformer_training(env_or_buffer)

    def _setup_decision_transformer_training(self, env_or_buffer):
        if isinstance(self._decision_transformer, DeterministicDecisionTransformer):
            trainer_config = MT.dt_trainers.DeterministicDecisionTransformerTrainerConfig(
                context_length=self._config.context_length)
            solvers = {self._decision_transformer.scope_name: self._decision_transformer_solver}
            wd_solver = self._decision_transformer_wd_solver
            wd_solvers = None if wd_solver is None else {self._decision_transformer.scope_name: wd_solver}
            decision_transformer_trainer = MT.dt_trainers.DeterministicDecisionTransformerTrainer(
                models=self._decision_transformer,
                solvers=solvers,
                wd_solvers=wd_solvers,
                env_info=self._env_info,
                config=trainer_config)
            return decision_transformer_trainer
        if isinstance(self._decision_transformer, StochasticDecisionTransformer):
            trainer_config = MT.dt_trainers.StochasticDecisionTransformerTrainerConfig(
                context_length=self._config.context_length)
            solvers = {self._decision_transformer.scope_name: self._decision_transformer_solver}
            wd_solver = self._decision_transformer_wd_solver
            wd_solvers = None if wd_solver is None else {self._decision_transformer.scope_name: wd_solver}
            decision_transformer_trainer = MT.dt_trainers.StochasticDecisionTransformerTrainer(
                models=self._decision_transformer,
                solvers=solvers,
                wd_solvers=wd_solvers,
                env_info=self._env_info,
                config=trainer_config)
            return decision_transformer_trainer
        raise NotImplementedError(
            'Unknown model type. Model should be either Deterministic/StochasticDecisionTransformer')

    def _run_online_training_iteration(self, env):
        raise NotImplementedError(f'Online training is not supported for {self.__name__}')

    def _run_offline_training_iteration(self, buffer):
        assert isinstance(buffer, TrajectoryReplayBuffer)
        self._decision_transformer_training(buffer, run_epoch=True)

    def _decision_transformer_training(self, replay_buffer, run_epoch):
        if run_epoch:
            buffer_iterator = _TrajectoryBufferIterator(
                replay_buffer, self._config.batch_size, self._config.context_length)
            # Run 1 epoch
            for trajectories, info in buffer_iterator:
                self._decision_transformer_iteration(trajectories, info)
        else:
            trajectories, info = replay_buffer.sample_trajectories_portion(self._config.batch_size,
                                                                           self._config.context_length)
            self._decision_transformer_iteration(trajectories, info)

    def _decision_transformer_iteration(self, trajectories, info):
        batch = None
        experiences = []
        for trajectory in trajectories:
            # trajectory: (s1, a1, r1, t1, s_next1, info1), ..., (sN, aN, rN, tN, s_nextN, infoN)
            # -> ([s1...sN] , [a1...aN], [r1...rN] , [t1...tN], [s_next1...s_nextN], [info1...infoN])
            marshaled = marshal_experiences(trajectory)
            experiences.append(marshaled)
        (s, a, _, _, _, extra, *_) = marshal_experiences(experiences)
        extra['target'] = a
        extra['rtg'] = extra['rtg']  # NOTE: insure that 'rtg' exists
        extra['timesteps'] = extra['timesteps'][:, 0:1, :]
        batch = TrainingBatch(batch_size=self._config.batch_size,
                              s_current=s,
                              a_current=a,
                              weight=info['weights'],
                              next_step_batch=None,
                              extra=extra)

        self._decision_transformer_trainer_state = self._decision_transformer_trainer.train(batch)
        if self._lr_scheduler is not None:
            # iteration_num is dummy
            new_learning_rate = self._lr_scheduler.get_learning_rate(self._iteration_num)
            self._decision_transformer_trainer.set_learning_rate(new_learning_rate)

    def _models(self):
        models = {}
        models[self._decision_transformer.scope_name] = self._decision_transformer
        if self._decision_transformer_wd_solver is not None:
            models[f'{self._decision_transformer.scope_name}_wd'] = self._decision_transformer
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._decision_transformer.scope_name] = self._decision_transformer_solver
        if self._decision_transformer_wd_solver is not None:
            solvers[f'{self._decision_transformer.scope_name}_wd'] = self._decision_transformer_wd_solver
        return solvers

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(
            env_or_env_info, gym.Env) else env_or_env_info
        return ((env_info.is_continuous_action_env() or env_info.is_discrete_action_env())
                and not env_info.is_tuple_action_env())

    @classmethod
    def is_rnn_supported(self):
        return False

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(DecisionTransformer, self).latest_iteration_state
        if hasattr(self, '_decision_transformer_trainer_state'):
            latest_iteration_state['scalar'].update(
                {'loss': float(self._decision_transformer_trainer_state['loss'])})
        return latest_iteration_state

    @property
    def trainers(self):
        return {"decision_transformer": self._decision_transformer_trainer}


class _TrajectoryBufferIterator(BufferIterator):
    _replay_buffer: TrajectoryReplayBuffer

    def __init__(self, buffer, batch_size, portion_length, shuffle=True, repeat=True):
        super().__init__(buffer, batch_size, shuffle, repeat)
        self._portion_length = portion_length

    def _sample(self, indices):
        return self._replay_buffer.sample_indices_portion(indices, portion_length=self._portion_length)
