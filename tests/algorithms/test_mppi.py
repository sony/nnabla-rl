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

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


class LinearDynamics(Dynamics):
    def __init__(self, dt=0.2):
        super().__init__()
        # state (position, velocity). velocity does not change without input
        self._A = np.array([[1, dt], [0, 1]])
        # input changes the velocity
        self._B = np.array([[0, 0], [0, dt]])

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        x_next = self._A.dot(x) + self._B.dot(u)
        return x_next, {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 2


class QuadraticCostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        self._Q = np.array([[1, 0], [0, 1]])
        self._R = np.array([[0.01, 0], [0, 0.01]])

    def evaluate(self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False) -> np.ndarray:
        if final_state:
            return x.T.dot(self._Q).dot(x) + u.T.dot(self._R).dot(u)
        else:
            return 0.0

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError


class TestMPPI(object):
    def test_algorithm_name(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        cost_function = QuadraticCostFunction()

        mppi = A.MPPI(env, cost_function=cost_function)

        assert mppi.__name__ == 'MPPI'

    def test_continuous_action_env_supported(self):
        '''
        Check that no error occurs when training on continuous action env
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        cost_function = QuadraticCostFunction()

        A.MPPI(env, cost_function=cost_function)

    def test_discrete_action_env_not_supported(self):
        '''
        Check that error occurs when training on discrete action env
        '''
        env = E.DummyDiscrete()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            A.MPPI(env, cost_function=cost_function)

    def test_compute_eval_action(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        cost_function = QuadraticCostFunction()

        T = 20
        mppi_config = A.MPPIConfig(T=T)
        mppi = A.MPPI(env, cost_function=cost_function, config=mppi_config)

        # initial pose
        x0 = np.array([[2.0], [0.0]])
        mppi_action = mppi.compute_eval_action(x0)

        assert mppi_action.shape == (*env.action_space.shape, )

    def test_compute_trajectory(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        T = 100
        covariance = np.eye(N=2) * 0.3
        config = A.MPPIConfig(T=T, use_known_dynamics=True, dt=0.2, covariance=covariance,)
        mppi = A.MPPI(env,
                      known_dynamics=dynamics,
                      cost_function=cost_function,
                      config=config)

        # initial pose
        x0 = np.array([[2.5], [0.0]])
        initial_trajectory = self._compute_initial_trajectory(x0, dynamics, T)
        trajectory, _ = mppi.compute_trajectory(initial_trajectory)

        # mppi just return dummy states.
        # simulate behavior again
        x = x0
        real_trajectory = []
        for t, (_, u) in enumerate(trajectory):
            x = np.squeeze(x)
            real_trajectory.append((x, u))
            x, *_ = dynamics.next_state(x, u, t)
        (pos, _) = real_trajectory[-2][0]
        assert np.abs(pos) < 0.5

    def test_run_online_training(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ), max_episode_steps=5)
        cost_function = QuadraticCostFunction()

        config = A.MPPIConfig(batch_size=2, training_iterations=2)
        mppi = A.MPPI(env, cost_function=cost_function, config=config)
        mppi.train_online(env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that error occurs when calling offline training
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ), max_episode_steps=5)
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            mppi = A.MPPI(env, cost_function=cost_function)
            mppi.train_offline(env, total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.MPPIConfig(M=-1, K=-1, T=-1, dt=-1.0)

    def _compute_initial_trajectory(self, x0, dynamics, T):
        trajectory = []
        x = x0
        for t in range(T):
            u = np.zeros(shape=(dynamics.action_dim(), 1))
            trajectory.append((x, u))
            x, _ = dynamics.next_state(x, u, t)
        return trajectory


if __name__ == "__main__":
    pytest.main()
