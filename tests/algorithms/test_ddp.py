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
    def __init__(self, dt=0.01):
        super().__init__()
        # state (position, velocity). velocity does not change without input
        self._A = np.array([[1, dt], [0, dt]])
        # input changes the velocity
        self._B = np.array([[0, 0], [0, dt]])

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._A.dot(x) + self._B.dot(u), {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._B

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state_dim = self.state_dim()
        action_dim = self.action_dim()
        Fxx = np.zeros(shape=(state_dim, state_dim, state_dim))
        Fxu = np.zeros(shape=(state_dim, state_dim, action_dim))
        Fux = np.zeros(shape=(state_dim, action_dim, state_dim))
        Fuu = np.zeros(shape=(state_dim, action_dim, action_dim))
        return Fxx, Fxu, Fux, Fuu

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 2


class QuadraticCostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        self._Q = np.array([[1, 0], [0, 1]])
        self._R = np.array([[10, 0], [0, 10]])
        self._F = np.array([[1, 1], [1, 1]])

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        if final_state:
            return x.T.dot(self._Q).dot(x)
        else:
            # Assuming that target state is zero
            return x.T.dot(self._Q).dot(x) + 2.0*x.T.dot(self._F).dot(u) + u.T.dot(self._R).dot(u)

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if final_state:
            return 2.0 * self._Q.dot(x), None
        else:
            return 2.0 * self._Q.dot(x) + 2.0 * self._F.dot(u), 2.0 * self._R.dot(u) + 2.0 * self._F.dot(x)

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if final_state:
            return self._Q, None, None, None, None
        else:
            return self._Q, self._F, self._F, self._R


class TestDDP(object):
    def test_algorithm_name(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        ddp = A.DDP(env, dynamics=dynamics, cost_function=cost_function)

        assert ddp.__name__ == 'DDP'

    def test_continuous_action_env_supported(self):
        '''
        Check that no error occurs when training on continuous action env
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        A.DDP(env, dynamics=dynamics, cost_function=cost_function)

    def test_discrete_action_env_not_supported(self):
        '''
        Check that error occurs when training on discrete action env
        '''
        env = E.DummyDiscrete()
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            A.DDP(env, dynamics=dynamics, cost_function=cost_function)

    def test_compute_eval_action(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        T = 20
        lqr_config = A.LQRConfig(T_max=T)
        lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function, config=lqr_config)

        ddp_config = A.DDPConfig(T_max=T, num_iterations=100)
        ddp = A.DDP(env, dynamics=dynamics, cost_function=cost_function, config=ddp_config)

        # initial pose
        x0 = np.array([[2.0], [0.0]])
        lqr_action = lqr.compute_eval_action(x0)
        ddp_action = ddp.compute_eval_action(x0)

        np.testing.assert_almost_equal(lqr_action, ddp_action, decimal=3)

    def test_compute_trajectory(self):
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        T = 20
        lqr_config = A.LQRConfig(T_max=T)
        lqr = A.LQR(env, dynamics=dynamics, cost_function=cost_function, config=lqr_config)

        ddp_config = A.DDPConfig(T_max=T, num_iterations=100)
        ddp = A.DDP(env, dynamics=dynamics, cost_function=cost_function, config=ddp_config)

        # initial pose
        x0 = np.array([[2.0], [0.0]])
        initial_trajectory = self._compute_initial_trajectory(x0, dynamics, T)
        lqr_trajectory, _ = lqr.compute_trajectory(initial_trajectory)
        ddp_trajectory, _ = ddp.compute_trajectory(initial_trajectory)

        for (lqr_state, ddp_state) in zip(lqr_trajectory[:-1], ddp_trajectory[:-1]):
            np.testing.assert_almost_equal(lqr_state[0], ddp_state[0], decimal=3)
            np.testing.assert_almost_equal(lqr_state[1], ddp_state[1], decimal=3)

    def test_run_online_training(self):
        '''
        Check that error occurs when calling online training
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            ddp = A.DDP(env, dynamics=dynamics, cost_function=cost_function)
            ddp.train_online(env)

    def test_run_offline_training(self):
        '''
        Check that error occurs when calling offline training
        '''
        env = E.DummyContinuous(observation_shape=(2, ), action_shape=(2, ))
        dynamics = LinearDynamics()
        cost_function = QuadraticCostFunction()

        with pytest.raises(Exception):
            ddp = A.DDP(env, dynamics=dynamics, cost_function=cost_function)
            ddp.train_offline(env)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.DDPConfig(T_max=-1)
        with pytest.raises(ValueError):
            A.DDPConfig(num_iterations=-1)

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
