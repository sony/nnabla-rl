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
        self._A = np.array([[1, dt], [0, dt]])
        # input changes the velocity
        self._B = np.array([[0, 0], [0, dt]])

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._A.dot(x) + self._B.dot(u), {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._B

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
        self._F = np.array([[1, 1], [1, 1]])

    def evaluate(self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False) -> np.ndarray:
        raise NotImplementedError

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

        if final_state:
            return self._Q, None, None, None, None
        else:
            return self._Q, self._F, self._F, self._R


class TestLQR(object):
    def test_algorithm_name(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function)

        assert lqr.__name__ == 'LQR'

    def test_continuous_action_env_supported(self):
        '''
        Check that no error occurs when training on continuous action env
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        A.LQR(env, dynamics=dynamics, cost_function=cost_function)

    def test_discrete_action_env_not_supported(self):
        '''
        Check that error occurs when training on discrete action env
        '''
        env = E.DummyDiscrete()
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            A.LQR(env, dynamics=dynamics, cost_function=cost_function)

    def test_compute_eval_action(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        T = 20
        lqr_config = A.LQRConfig(T_max=T)
        lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function, config=lqr_config)

        # initial pose
        x0 = np.array([[2.0], [0.0]])
        lqr_action = lqr.compute_eval_action(x0)

        assert lqr_action.shape == (*env.action_space.shape, 1)

    def test_compute_trajectory(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        T = 100
        config = A.LQRConfig(T_max=T)
        lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function, config=config)

        # initial pose
        x0 = np.array([[2.5], [0.0]])
        initial_trajectory = self._compute_initial_trajectory(x0, dynamics, T)
        trajectory, _ = lqr.compute_trajectory(initial_trajectory)

        # position and velocity should be zero after optimal control
        (pos, vel) = trajectory[-2]
        np.testing.assert_almost_equal(pos, 0.0, decimal=4)
        np.testing.assert_almost_equal(vel, 0.0, decimal=4)

    def test_run_online_training(self):
        '''
        Check that error occurs when calling online training
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function)
            lqr.train_online(env)

    def test_run_offline_training(self):
        '''
        Check that error occurs when calling offline training
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function)
            lqr.train_offline(env)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.LQRConfig(T_max=-1)

    def _compute_initial_trajectory(self, x0, dynamics, T):
        trajectory = []
        x = x0
        for t in range(T - 1):
            u = np.zeros(shape=(dynamics.action_dim(), 1))
            trajectory.append((x, u))
            x, _ = dynamics.next_state(x, u, t)
        # append final state
        trajectory.append((x, None))
        return trajectory


if __name__ == "__main__":
    pytest.main()
