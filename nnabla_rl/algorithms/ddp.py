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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import gym
import numpy as np

from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


@dataclass
class DDPConfig(AlgorithmConfig):
    '''
    List of configurations for DDP (Differential Dynamic Programming) algorithm

    Args:
        T_max (int): Planning time step length. Defaults to 50.
        num_iterations (int): Number of iterations for the optimization. Defaults to 10.
        mu_min (float): Minimum value for regularizing the hessian of the value funtion. Defaults to 1e-6.
        modification_factor (float): Modification factor for the regularizer. Defaults to 2.0.
        accept_improvement_ratio (float): Threshold value for deciding to accept the update or not. Defaults to 0.0
    '''
    T_max: int = 50
    num_iterations: int = 10
    mu_min: float = 1e-6
    modification_factor: float = 2.0
    accept_improvement_ratio: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        self._assert_positive(self.T_max, 'T_max')
        self._assert_positive(self.num_iterations, 'num_iterations')


class DDP(Algorithm):
    '''Differential Dynamic Programming algorithm.
    This class implements the differential dynamic programming (DDP) algorithm proposed by D. Mayne in the paper:
    "A Second-order Gradient Method for Determining Optimal Trajectories of Non-linear Discrete-time Systems".
    We also referred the paper written by Y. Tassa et al.:
    "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    for the implementation of this algorithm.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        dynamics (:py:class:`Dynamics <nnabla_rl.non_nn_models.dynamics.Dynamics>`):
            dynamics of the system to control
        cost_function (:py:class:`Dynamics <nnabla_rl.non_nn_models.cost_function.CostFunction>`):
            cost function to optimize the trajectory
        config (:py:class:`DDPConfig <nnabla_rl.algorithmss.ilqr.DDPConfig>`):
            the parameter for DDP controller
    '''
    _config: DDPConfig

    def __init__(self,
                 env_or_env_info,
                 dynamics: Dynamics,
                 cost_function: CostFunction,
                 config=DDPConfig()):
        super(DDP, self).__init__(env_or_env_info, config=config)
        self._dynamics = dynamics
        self._cost_function = cost_function

    @eval_api
    def compute_eval_action(self, state, *, begin_of_episode=False):
        dynamics = self._dynamics
        cost_function = self._cost_function
        x0 = state
        u0 = [np.zeros((dynamics.action_dim(), 1)) for t in range(self._config.T_max - 1)]
        initial_trajectory = self._compute_initial_trajectory(x0, dynamics, self._config.T_max, u0)
        improved_trajectory, _ = self._optimize(initial_trajectory, dynamics, cost_function)

        return improved_trajectory[0][1]

    @eval_api
    def compute_trajectory(self,
                           initial_trajectory:  Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]) \
            -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        assert len(initial_trajectory) == self._config.T_max
        dynamics = self._dynamics
        cost_function = self._cost_function
        mu = 0.0
        delta = 0.0
        trajectory = initial_trajectory
        for _ in range(self._config.num_iterations):
            trajectory, trajectory_info, mu, delta = \
                self._improve_trajectory(trajectory, dynamics, cost_function, mu, delta)

        return trajectory, trajectory_info

    def _optimize(self,
                  initial_state: Union[np.ndarray, Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]],
                  dynamics: Dynamics,
                  cost_function: CostFunction,
                  **kwargs) \
            -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        assert len(initial_state) == self._config.T_max
        initial_state = cast(Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], initial_state)
        mu = 0.0
        delta = 0.0
        trajectory = initial_state
        for _ in range(self._config.num_iterations):
            trajectory, trajectory_info, mu, delta = \
                self._improve_trajectory(trajectory, dynamics, cost_function, mu, delta)

        return trajectory, trajectory_info

    def _compute_initial_trajectory(self, x0, dynamics, T, u):
        trajectory = []
        x = x0
        for t in range(T - 1):
            trajectory.append((x, u[t]))
            x, _ = dynamics.next_state(x, u[t], t)
        trajectory.append((x, None))
        return trajectory

    def _improve_trajectory(self,
                            trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]],
                            dynamics: Dynamics,
                            cost_function: CostFunction,
                            mu: float,
                            delta: float) -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]],
                                                   Sequence[Dict[str, Any]],
                                                   float,
                                                   float]:
        while True:
            ks, Ks, Qus, Quus, Quu_invs, success = self._backward_pass(trajectory, dynamics, cost_function, mu)
            mu, delta = self._update_regularizer(mu, delta, increase=not success)
            if success:
                break

        # Backtracking linear search
        alphas = 0.9**(np.arange(10) ** 2)
        improved_trajectory = trajectory
        improved_trajectory_info: Sequence[Dict[str, Any]] = []
        J_current = self._compute_cost(trajectory, cost_function)
        for alpha in alphas:
            new_trajectory, new_trajectory_info = self._forward_pass(trajectory, dynamics, ks, Ks, alpha)
            J_new = self._compute_cost(new_trajectory, cost_function)
            delta_J = self._expected_cost_reduction(ks, Qus, Quus, alpha)

            reduction_ratio = (J_current - J_new) / np.abs(delta_J)
            if self._config.accept_improvement_ratio < reduction_ratio:
                improved_trajectory = new_trajectory
                # append Quu
                for info, k, K, Quu, Quu_inv in zip(new_trajectory_info, ks, Ks, Quus, Quu_invs):
                    info.update({'k': k, 'K': K, 'Quu': Quu, 'Quu_inv': Quu_inv})
                improved_trajectory_info = new_trajectory_info
                break
        return improved_trajectory, improved_trajectory_info, mu, delta

    def _update_regularizer(self, mu, delta, increase):
        if increase:
            # increase mu
            delta0 = self._config.modification_factor
            delta = max(delta0, delta * delta0)
            mu = max(self._config.mu_min, mu * delta)
        else:
            # decrease mu
            delta0 = self._config.modification_factor
            delta = min(1.0 / delta0, delta / delta0)
            mu = mu * delta
            if mu < self._config.mu_min:
                mu = 0.0
        return mu, delta

    def _backward_pass(self, trajectory, dynamics, cost_function, mu):
        x_last, u_last = trajectory[-1]
        # Initialize Vx and Vxx to the gradient/hessian of value function of the final state of the trajectory
        Vx, *_ = cost_function.gradient(x_last, u_last, self._config.T_max, final_state=True)
        Vxx, *_ = cost_function.hessian(x_last, u_last, self._config.T_max, final_state=True)
        E = np.identity(n=Vxx.shape[0])

        ks: List[np.ndarray] = []
        Ks: List[np.ndarray] = []
        Qus: List[np.ndarray] = []
        Quus: List[np.ndarray] = []
        Quu_invs: List[np.ndarray] = []
        x_dim = dynamics.state_dim()
        u_dim = dynamics.action_dim()
        for t in reversed(range(self._config.T_max - 1)):
            (x, u) = trajectory[t]
            Cx, Cu = cost_function.gradient(x, u, self._config.T_max - t - 1)
            Cxx, Cxu, Cux, Cuu = cost_function.hessian(x, u, self._config.T_max - t - 1)

            Fx, Fu = dynamics.gradient(x, u, self._config.T_max - t - 1)
            # Hessians should be a 3d tensor
            Fxx, Fxu, Fux, Fuu = dynamics.hessian(x, u,  self._config.T_max - t - 1)

            Quu = Cuu + Fu.T.dot(Vxx + mu * E).dot(Fu) + np.tensordot(Vx, Fuu, axes=(0, 0)).reshape((u_dim, u_dim))

            if not self._is_positive_definite(Quu):
                return ks, Ks, Qus, Quus, Quu_invs, False

            Qx = Cx + Fx.T.dot(Vx)
            Qu = Cu + Fu.T.dot(Vx)

            Qxx = Cxx + Fx.T.dot(Vxx).dot(Fx) + np.tensordot(Vx, Fxx, axes=(0, 0)).reshape((x_dim, x_dim))
            # NOTE: Qxu and Qux should be symmetric and same matrix. i.e. Qxu = Qux and Qxu.T = Qux
            Qxu = Cxu + Fu.T.dot(Vxx + mu * E).dot(Fx).T + np.tensordot(Vx, Fxu, axes=(0, 0)).reshape((x_dim, u_dim))
            Qux = Cux + Fu.T.dot(Vxx + mu * E).dot(Fx) + np.tensordot(Vx, Fux, axes=(0, 0)).reshape((u_dim, x_dim))
            assert np.allclose(Qxu, Qux.T)

            Quu_inv = np.linalg.inv(Quu)
            k = -Quu_inv.dot(Qu)
            K = -Quu_inv.dot(Qux)

            ks.append(k)
            Ks.append(K)
            Qus.append(Qu)
            Quus.append(Quu)
            Quu_invs.append(Quu_inv)

            Vx = Qx + K.T.dot(Quu).dot(k) + K.T.dot(Qu) + Qux.T.dot(k)
            Vxx = Qxx + K.T.dot(Quu).dot(K) + K.T.dot(Qux) + Qux.T.dot(K)

        ks = list(reversed(ks))
        Ks = list(reversed(Ks))
        Qus = list(reversed(Qus))
        Quus = list(reversed(Quus))
        Quu_invs = list(reversed(Quu_invs))

        return ks, Ks, Qus, Quus, Quu_invs, True

    def _forward_pass(
        self,
        trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]],
        dynamics: Dynamics,
        ks: List[np.ndarray],
        Ks: List[np.ndarray],
        alpha: float
    ) -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        x_hat = trajectory[0][0]
        new_trajectory = []
        new_trajectory_info: List[Dict[str, Any]] = []
        for t, ((x, u), k, K) in enumerate(zip(trajectory[:-1], ks, Ks)):  # not include final step
            assert u is not None
            dx = x_hat - x
            du = alpha * k + K.dot(dx)
            u_hat = u + du
            new_trajectory.append((x_hat, u_hat))
            new_trajectory_info.append({})

            x_hat, _ = dynamics.next_state(x_hat, u_hat, t)
        new_trajectory.append((x_hat, None))  # final timestep input is None
        new_trajectory_info.append({})
        return new_trajectory, new_trajectory_info

    def _compute_cost(
        self, trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], cost_function: CostFunction
    ) -> float:
        J = 0.0
        for t, (x, u) in enumerate(trajectory[:-1]):  # not include final step
            J += float(cost_function.evaluate(x, u, t))

        J += float(cost_function.evaluate(trajectory[-1][0], trajectory[-1][1], len(trajectory), final_state=True))
        return J

    def _expected_cost_reduction(self, ks, Qus, Quus, alpha) -> float:
        delta_J = 0.0
        for (k, Qu, Quu) in zip(ks, Qus, Quus):
            linear_part = alpha * k.T.dot(Qu)
            squared_part = 0.5 * (alpha ** 2.0) * k.T.dot(Quu).dot(k)
            delta_J += float(linear_part) + float(squared_part)
        return delta_J

    def _is_positive_definite(self, symmetric_matrix: np.ndarray):
        return np.all(np.linalg.eigvals(symmetric_matrix) > 0.0)

    def _before_training_start(self, env_or_buffer):
        raise NotImplementedError('You do not need training to use this algorithm.')

    def _run_online_training_iteration(self, env):
        raise NotImplementedError('You do not need training to use this algorithm.')

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError('You do not need training to use this algorithm.')

    def _after_training_finish(self, env_or_buffer):
        raise NotImplementedError('You do not need training to use this algorithm.')

    def _models(self):
        return {}

    def _solvers(self):
        return {}

    @classmethod
    def is_supported_env(cls, env_or_env_info):
        env_info = EnvironmentInfo.from_env(env_or_env_info) if isinstance(env_or_env_info, gym.Env) \
            else env_or_env_info
        return not env_info.is_discrete_action_env()

    @property
    def trainers(self):
        return {}
