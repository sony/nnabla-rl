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
class LQRConfig(AlgorithmConfig):
    '''
    List of configurations for LQR (Linear Quadratic Regulator) algorithm

    Args:
        T_max (int): Planning time step length. Defaults to 50.
    '''
    T_max: int = 50

    def __post_init__(self):
        super().__post_init__()

        self._assert_positive(self.T_max, 'T_max')


class LQR(Algorithm):
    '''LQR (Linear Quadratic Regulator) algorithm.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        dynamics (:py:class:`Dynamics <nnabla_rl.non_nn_models.dynamics.Dynamics>`):
            dynamics of the system to control
        cost_function (:py:class:`Dynamics <nnabla_rl.non_nn_models.cost_function.CostFunction>`):
            cost function to optimize the trajectory
        config (:py:class:`LQRConfig <nnabla_rl.algorithmss.lqr.LQRConfig>`):
            the parameter for LQR controller
    '''
    _config: LQRConfig

    def __init__(self,
                 env_or_env_info,
                 dynamics: Dynamics,
                 cost_function: CostFunction,
                 config=LQRConfig()):
        super(LQR, self).__init__(env_or_env_info, config=config)
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
                           initial_trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]) \
            -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        assert len(initial_trajectory) == self._config.T_max
        dynamics = self._dynamics
        cost_function = self._cost_function
        return self._optimize(initial_trajectory, dynamics, cost_function)

    def _compute_initial_trajectory(self, x0, dynamics, T, u):
        trajectory = []
        x = x0
        for t in range(T - 1):
            trajectory.append((x, u[t]))
            x, _ = dynamics.next_state(x, u[t], t)
        trajectory.append((x, None))
        return trajectory

    def _optimize(self,
                  initial_state: Union[np.ndarray, Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]],
                  dynamics: Dynamics,
                  cost_function: CostFunction,
                  **kwargs) \
            -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        assert len(initial_state) == self._config.T_max
        initial_state = cast(Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], initial_state)
        x_last, u_last = initial_state[-1]
        Sk, *_ = cost_function.hessian(x_last, u_last, self._config.T_max, final_state=True)

        matrices: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for t in reversed(range(self._config.T_max - 1)):
            (x, u) = initial_state[t]
            assert u is not None
            A, B = dynamics.gradient(x, u, self._config.T_max - t - 1)
            assert B is not None
            Q, F, _, R = cost_function.hessian(x, u, self._config.T_max - t - 1)
            assert F is not None
            assert R is not None
            C = np.linalg.inv(R + (B.T.dot(Sk).dot(B)))
            D = (F.T + B.T.dot(Sk).dot(A))
            Sk = Q + A.T.dot(Sk).dot(A) - D.T.dot(C).dot(D)
            matrices.append((Sk, A, B, R, F))

        trajectory: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        trajectory_info: List[Dict[str, np.ndarray]] = []
        x = initial_state[0][0]
        for t, (S, A, B, R, F) in enumerate(reversed(matrices)):
            u = self._compute_optimal_input(x, S, A, B, R, F)
            trajectory.append((x, u))
            # Save quadratic cost coefficient R as Quu and R^-1 as Quu_inv
            trajectory_info.append({'Quu': R, 'Quu_inv': np.linalg.inv(R)})
            x, _ = dynamics.next_state(x, u, t)

        trajectory.append((x, None))  # final timestep input is None
        trajectory_info.append({})
        return trajectory, trajectory_info

    def _compute_optimal_input(self, x, S, A, B, R, F) -> np.ndarray:
        C = np.linalg.inv(R + (B.T.dot(S).dot(B)))
        D = (F.T + B.T.dot(S).dot(A))
        return cast(np.ndarray, -C.dot(D).dot(x))

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
