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
import argparse
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from nnabla_rl.algorithms.ilqr import iLQR, iLQRConfig
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


class AnglePendulum(PendulumEnv):
    '''Pendulum environment where the state is angle and angle speed.
    '''

    def __init__(self, g=10):
        super().__init__(g=g)

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([angle_normalize(theta), thetadot], dtype=np.float32)


class PendulumDynamics(Dynamics):
    def __init__(self):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.length = 1.0

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        if batched:
            raise NotImplementedError
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        th = x.flatten()[0]  # th := theta
        thdot = x.flatten()[1]
        u = u.flatten()[0]
        newthdot = thdot + (3 * self.g / (2 * self.length) * np.sin(th) + 3.0 / (self.m * self.length**2) * u) * self.dt
        newth = th + newthdot * self.dt
        return np.array([[newth], [newthdot]], dtype=np.float32), {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if batched:
            raise NotImplementedError
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        th = x.flatten()[0]  # th := theta
        Fx = np.zeros((self.state_dim(), self.state_dim()))

        Fx[0, 0] = 1.
        Fx[0, 1] = 1. * self.dt
        Fx[1, 0] = 3 * self.g / (2 * self.length) * np.cos(th) * self.dt
        Fx[1, 1] = 1.

        Fu = np.zeros((self.state_dim(), self.action_dim()))
        Fu[1, 0] = (3.0 / (self.m * self.length**2)) * self.dt

        return Fx, Fu

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 1


class PendulumCostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        self._weight_u = 1.
        self._weight_th = 2.5
        self._weight_thdot = 0.5

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        if batched:
            raise NotImplementedError
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        th = x.flatten()[0]  # th := theta
        thdot = x.flatten()[1]
        state_cost = self._weight_th * self._angle_normalize(th) ** 2 + self._weight_thdot * thdot**2

        if final_state:
            return state_cost
        else:
            act_cost = self._weight_u * (u.flatten()**2)
            return state_cost + act_cost

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        th = x.flatten()[0]  # th := theta
        thdot = x.flatten()[1]
        Cx = np.array([[2 * self._weight_th * self._angle_normalize(th)], [2. * self._weight_thdot * thdot]])
        if final_state:
            return (Cx, None)
        else:
            Cu = 2 * self._weight_u * u
            return (Cx, Cu)

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        x_dim = x.shape[0]
        Cxx = np.diag([2.0 * self._weight_th, 2.0 * self._weight_thdot])

        if final_state:
            return Cxx, None, None, None
        else:
            u_dim = u.shape[0]
            Cux = np.zeros((u_dim, x_dim))
            Cxu = np.zeros((x_dim, u_dim))
            return Cxx, Cxu, Cux, 2.0 * np.eye(1) * self._weight_u

    def _angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


def compute_initial_trajectory(x0, dynamics, T, u):
    trajectory = []
    x = x0
    for t in range(T - 1):
        trajectory.append((x, u[t]))
        x, _ = dynamics.next_state(x, u[t], t)
    trajectory.append((x, None))
    return trajectory


def run_control(args):
    env = AnglePendulum()
    dynamics = PendulumDynamics()
    cost_function = PendulumCostFunction()

    config = iLQRConfig(T_max=args.T, num_iterations=50)
    ilqr = iLQR(env, dynamics, cost_function, config)

    for _ in range(args.num_episodes):
        state = env.reset()
        state = np.reshape(state, (state.shape[0], 1))
        done = False
        total_reward = 0
        initial_u = [np.zeros((dynamics.action_dim(), 1)) for t in range(args.T - 1)]
        while not done:
            initial_trajectory = compute_initial_trajectory(state, dynamics, args.T, initial_u)
            start = time.time()
            improved_trajectory, trajectory_info = ilqr.compute_trajectory(initial_trajectory)
            end = time.time()
            print(f'optimization time: {end - start} [s]')

            u = improved_trajectory[0][1].reshape((1, ))
            next_state, reward, done, *_ = env.step(u)
            total_reward += reward

            if args.render:
                env.render()

            state = next_state
            initial_u = [x_u[1] for x_u in improved_trajectory[1:-1]]
            initial_u.append(np.zeros((dynamics.action_dim(), 1)))

        print(total_reward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--T', type=int, default=25)
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()
    run_control(args)


if __name__ == '__main__':
    main()
