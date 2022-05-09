# Copyright 2022 Sony Group Corporation.
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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.model_trainers as MT
import nnabla_rl.random as rl_random
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.algorithms.common_utils import _DeterministicStatePredictor
from nnabla_rl.builders import ModelBuilder, ReplayBufferBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import DeterministicDynamics, MPPIDeterministicDynamics
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import context
from nnabla_rl.utils.data import marshal_experiences, unzip


@dataclass
class MPPIConfig(AlgorithmConfig):
    '''
    List of configurations for MPPI (Model Predictive Path Integral) algorithm

    Args:
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.001.
        batch_size(int): training batch size. Defaults to 100.
        replay_buffer_size (int): capacity of the replay buffer. Defaults to 1000000.
        training_iterations (int): dynamics training iterations. Defaults to 500.
        lmb (float): scalar variable lambda used in the difinision of free-energy.
        M (int): number of trials per training iteration. Defaults to 1.
        K (int): number of samples for importance sampling. Defaults to 100.
        T (int): number of prediction steps. Defaults to 100.
        covariance (np.ndarray): Covariance of gaussian noise applied to control inputs.
            If covariance is not specified, covariance with unit variance will be used.
            Defaults to None.
        use_known_dynamics (bool): Use the dynamics model passed to the MPPI algorithm instead of trained model
            to compute actions.
        unroll_steps (int): Number of steps to unroll dynamics's tranining network.\
            The network will be unrolled even though the provided model doesn't have RNN layers.\
            Defaults to 1.
        burn_in_steps (int): Number of burn-in steps to initiaze dynamics's recurrent layer states during training.\
            This flag does not take effect if given model is not an RNN model.\
            Defaults to 0.
        reset_rnn_on_terminal (bool): Reset recurrent internal states to zero during training if episode ends. \
            This flag does not take effect if given model is not an RNN model. \
            Defaults to False.
        dt (float): Time interval between states. Defaults to 0.05 [s].
            We strongly recommended to adjust this interval considering the sensor frequency.
    '''
    learning_rate: float = 1.0*1e-3
    batch_size: int = 100
    replay_buffer_size: int = 1000000
    training_iterations: int = 500
    lmb: float = 1.0
    M: int = 1
    K: int = 500
    T: int = 100
    covariance: Optional[np.ndarray] = None
    use_known_dynamics: bool = False
    unroll_steps: int = 1
    burn_in_steps: int = 0
    reset_rnn_on_terminal: bool = False
    dt: float = 0.05

    def __post_init__(self):
        super().__post_init__()

        self._assert_positive(self.lmb, 'lmb')
        self._assert_positive(self.M, 'M')
        self._assert_positive(self.K, 'K')
        self._assert_positive(self.T, 'T')
        self._assert_positive(self.dt, 'dt')


class DefaultDynamicsBuilder(ModelBuilder[DeterministicDynamics]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: MPPIConfig,
                    **kwargs) -> DeterministicDynamics:
        return MPPIDeterministicDynamics(scope_name, dt=algorithm_config.dt)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: MPPIConfig,
                     **kwargs) -> nn.solver.Solver:
        # return NS.RMSprop(lr=algorithm_config.learning_rate)
        return NS.Adam(alpha=algorithm_config.learning_rate)


class DefaultReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self,  # type: ignore[override]
                            env_info: EnvironmentInfo,
                            algorithm_config: MPPIConfig,
                            **kwargs) -> ReplayBuffer:
        return ReplayBuffer(capacity=algorithm_config.replay_buffer_size)


class MPPI(Algorithm):
    '''MPPI (Model Predictive Path Integral) algorithm.
    This class implements the model predictive path integral (MPPI) algorithm
    proposed by G. Williams, et al. in the paper:
    "Information Theoretic MPC for Model-Based Reinforcement Learning"
    For details see: https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf

    Our implementation of MPPI assumes that environment's state consists of elements in the following order.
    :math:`(x_1, x_2, \\cdots, x_n, \\frac{dx_1}{dt}, \\frac{dx_2}{dt}, \\cdots, \\frac{dx_n}{dt})`.
    For example if you have two variables :math:`x` and :math:`\\theta`, then the state should be.
    :math:`(x, \\theta, \\dot{x}, \\dot{\\theta})`
    and not
    :math:`(x, \\dot{x}, \\theta, \\dot{\\theta})`.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        cost_function (:py:class:`CostFunction <nnabla_rl.numpy_models.cost_function.CostFunction>`):
            cost function to optimize the trajectory
        known_dynamics (:py:class:`Dynamics <nnabla_rl.numpy_models.dynamics.Dynamics>`):
            Dynamics model of target system to control.
            If this argument is not None, the algorithm will use the given dynamics model to compute the control input
            when compute_eval_action and compute_trajectory is called.
            This argument is optional. Defaults to None.
        state_normalizer (`Optional[Callable[[np.ndarray], np.ndarray]]`):
            Optional. State normalizing function is used to normalize state predicted
            state values to fit in proper range. For example you can provide state normalizer
            to fit :math:`\\theta` in :math:`-\\pi\\leq\\theta\\leq\\pi`

            Default is None.
        config (:py:class:`MPPIConfig <nnabla_rl.algorithmss.lqr.MPPIConfig>`):
            the parameter for MPPI controller
        dynamics_builder (:py:class:`ModelBuilder[DeterministicDynamics] <nnabla_rl.builders.ModelBuilder>`):
            builder of deterministic dynamics models
        dynamics_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`):
            builder of dynamics solvers
        replay_buffer_builder (:py:class:`ReplayBufferBuilder <nnabla_rl.builders.ReplayBufferBuilder>`):
            builder of replay_buffer. If you have bootstrap data, override the default builder
            and return a replay buffer with bootstrap data.
    '''
    _config: MPPIConfig
    _evaluation_dynamics: _DeterministicStatePredictor

    def __init__(self,
                 env_or_env_info,
                 cost_function: CostFunction,
                 known_dynamics: Optional[Dynamics] = None,
                 state_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 config: MPPIConfig = MPPIConfig(),
                 dynamics_builder: ModelBuilder[DeterministicDynamics] = DefaultDynamicsBuilder(),
                 dynamics_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 replay_buffer_builder: ReplayBufferBuilder = DefaultReplayBufferBuilder()):
        super(MPPI, self).__init__(env_or_env_info, config=config)
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            self._known_dynamics = known_dynamics
            self._dynamics = dynamics_builder('dynamics', env_info=self._env_info, algorithm_config=self._config)
            self._dynamics_solver = dynamics_solver_builder(env_info=self._env_info, algorithm_config=self._config)
            self._replay_buffer = replay_buffer_builder(env_info=self._env_info, algorithm_config=self._config)

        if self._config.use_known_dynamics:
            assert self._known_dynamics is not None

        self._cost_function = cost_function
        self._state_normalizer = state_normalizer
        self._evaluation_dynamics = _DeterministicStatePredictor(self._env_info, self._dynamics.shallowcopy())

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False):
        x = state
        u = np.asarray([np.zeros((self._env_info.action_dim, 1)) for _ in range(self._config.T)])
        _, control_inputs = self._compute_control_inputs(x, u)

        return control_inputs[0]

    @eval_api
    def compute_trajectory(self,
                           initial_trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]) \
            -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        assert len(initial_trajectory) == self._config.T
        x, u = unzip(initial_trajectory)

        dummy_states, control_inputs = self._compute_control_inputs(x[0], np.asarray(u))
        info: Sequence[Dict[str, Any]] = [{}] * len(control_inputs)
        return list(zip(dummy_states, control_inputs)), info

    def _compute_initial_trajectory(self, x0, dynamics, T, u):
        trajectory = []
        x = x0
        for t in range(T - 1):
            trajectory.append((x, u[t]))
            x, _ = dynamics.next_state(x, u[t], t)
        trajectory.append((x, None))
        return trajectory

    def _before_training_start(self, env_or_buffer):
        # set context globally to ensure that the training runs on configured gpu
        context.set_nnabla_context(self._config.gpu_id)
        self._dynamics_trainer = self._setup_dynamics_training(env_or_buffer)

    def _setup_dynamics_training(self, env_or_buffer):
        dynamics_trainer_config = MT.dynamics_trainers.MPPIDynamicsTrainerConfig(
            unroll_steps=self._config.unroll_steps,
            burn_in_steps=self._config.burn_in_steps,
            reset_on_terminal=self._config.reset_rnn_on_terminal,
            dt=self._config.dt)

        dynamics_trainer = MT.dynamics_trainers.MPPIDynamicsTrainer(
            models=self._dynamics,
            solvers={self._dynamics.scope_name: self._dynamics_solver},
            env_info=self._env_info,
            config=dynamics_trainer_config)
        return dynamics_trainer

    def _run_online_training_iteration(self, env):
        if self._config.batch_size < len(self._replay_buffer):
            for _ in range(self._config.training_iterations):
                self._mppi_training(self._replay_buffer)
        for _ in range(self._config.M):
            experiences = self._run_mppi(env)  # Dj
            self._replay_buffer.append_all(experiences)  # D U Dj

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError('You can not train MPPI only with buffer. Try online training.')

    def _mppi_training(self, replay_buffer):
        # train the dynamics model
        num_steps = self._config.burn_in_steps + self._config.unroll_steps
        experiences_tuple, info = replay_buffer.sample(self._config.batch_size, num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = (experiences_tuple, )
        assert len(experiences_tuple) == num_steps
        batch = None
        for experiences in reversed(experiences_tuple):
            (s, a, _, non_terminal, s_next, *_) = marshal_experiences(experiences)
            batch = TrainingBatch(batch_size=self._config.batch_size,
                                  s_current=s,
                                  a_current=a,
                                  s_next=s_next,
                                  non_terminal=non_terminal,
                                  weight=info['weights'],
                                  next_step_batch=batch,
                                  rnn_states={})

        self._dynamics_trainer_state = self._dynamics_trainer.train(batch)

    def _run_mppi(self, env):
        x = env.reset()
        control_inputs = np.zeros(shape=(self._config.T, self._env_info.action_dim))
        done = False
        experience = []
        while not done:
            _, improved_inputs = self._compute_control_inputs(x, control_inputs)
            u = improved_inputs[0]
            x_next, reward, done, *_ = env.step(u)
            non_terminal = 0.0 if done else 1.0
            experience.append((x, u, reward, non_terminal, x_next, {}))
            improved_inputs[0:-1] = improved_inputs[1:]
            control_inputs = improved_inputs
            x = x_next
        return experience

    def _compute_control_inputs(self, x, control_inputs):
        x = np.broadcast_to(x, shape=(self._config.K, *x.shape))
        if len(x.shape) == 3:
            x = np.squeeze(x, axis=-1)
        if len(control_inputs.shape) == 3:
            control_inputs = np.squeeze(control_inputs, axis=-1)
        dummy_states = []
        improved_inputs = control_inputs.copy()
        control_inputs = np.broadcast_to(control_inputs, shape=(self._config.K, *control_inputs.shape))
        mean = np.zeros(shape=(self._env_info.action_dim, ))
        cov = np.eye(N=self._env_info.action_dim)
        if self._config.covariance is not None:
            assert cov.shape == self._config.covariance.shape
            cov = self._config.covariance

        input_noise = rl_random.drng.multivariate_normal(mean, cov=cov, size=(self._config.K, self._config.T))
        S = np.zeros(shape=(self._config.K, 1))
        zero_control = np.zeros(shape=self._env_info.action_shape)
        batch_cov = np.broadcast_to(cov, shape=(self._config.K, *cov.shape))
        for t in range(self._config.T):
            dummy_states.append(x[0])
            u = control_inputs[:, t, :]
            e = input_noise[:, t, :]
            x_next = self._compute_next_state(x, u + e, t)
            if self._cost_function.support_batch():
                q_xt = self._cost_function.evaluate(x_next, zero_control, t, batched=True)
                S += q_xt
            else:
                for k in range(self._config.K):
                    q_xt = self._cost_function.evaluate(x_next[k], zero_control, t)
                    S[k] += q_xt
            S += self._config.lmb * (u[:, None, :] @ batch_cov @ e[:, :, None]).squeeze(axis=-1)

            x = x_next

        if self._cost_function.support_batch():
            S += self._cost_function.evaluate(x, zero_control, self._config.T, final_state=True, batched=True)
        else:
            for k in range(self._config.K):
                S[k] += self._cost_function.evaluate(x[k], zero_control, self._config.T, final_state=True)
        beta = np.min(S)
        eta = np.sum(np.exp(-(S - beta)/self._config.lmb))
        weights = np.exp(-(S - beta)/self._config.lmb) / eta

        du = np.sum(weights[:, np.newaxis, :] * input_noise, axis=0)
        improved_inputs += du
        # NOTE: clipping is important (if exist limits).
        improved_inputs = np.clip(improved_inputs, self._env_info.action_low, self._env_info.action_high)
        return dummy_states, improved_inputs

    def _compute_next_state(self, x: np.ndarray, u: np.ndarray, t: int):
        if self._known_dynamics is not None and self._config.use_known_dynamics:
            if self._known_dynamics.support_batch():
                x_next, *_ = self._known_dynamics.next_state(x, u, t, batched=True)
            else:
                x_next = np.empty(shape=x.shape)
                for k, (xk, uk) in enumerate(zip(x, u)):
                    next_state, *_ = self._known_dynamics.next_state(xk, uk, t)
                    x_next[k] = next_state.squeeze()
            if self._state_normalizer is not None:
                x_next = self._state_normalizer(x_next)
            return x_next
        else:
            u = np.clip(u, self._env_info.action_low, self._env_info.action_high)
            x_next, *_ = self._evaluation_dynamics(x, u, begin_of_episode=(t == 0))
            if self._state_normalizer is not None:
                x_next = self._state_normalizer(x_next)
            return x_next

    def _models(self):
        models = {}
        models[self._dynamics.scope_name] = self._dynamics
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._dynamics.scope_name] = self._dynamics_solver
        return solvers

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return not env_info.is_discrete_action_env()

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(MPPI, self).latest_iteration_state
        if hasattr(self, '_dynamics_trainer_state'):
            print('latest iteration state')
            latest_iteration_state['scalar'].update(
                {'dynamics_loss': float(self._dynamics_trainer_state['dynamics_loss'])})
        return latest_iteration_state

    @classmethod
    def is_rnn_supported(self):
        return True

    @property
    def trainers(self):
        return {"dynamics_trainer": self._dynamics_trainer}
