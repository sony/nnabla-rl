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
from gym import spaces
from gym.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap

from nnabla_rl.algorithms.ilqr import iLQR, iLQRConfig
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


class ContinuousAcrobot(AcrobotEnv):
    '''
    Continuous Acrobot environment.
    NOTE: This environment does not limit the input torque.
    '''

    def __init__(self):
        super().__init__()
        self.dt = 0.02
        high = np.array(
            [np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def step(self, torque):
        s = self.state

        x_and_u = np.append(s, torque)

        ns = rk4(self._dsdt, x_and_u, [0, self.dt])

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return (self._get_ob(), reward, terminal, {})

    def _terminal(self):
        return False

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1], s[2], s[3]], dtype=np.float32)


class AcrobotDynamics(Dynamics):
    def __init__(self):
        super().__init__()
        # link1
        self.m1 = 1.0  # [kg] mass of link1
        self.I1 = 1.0  # [kg*m^2] inertia around the center of mass of link 1
        self.l1 = 1.0  # [m] length of link1
        self.lc1 = 0.5  # [m] center of mass position of link1

        # link2
        self.m2 = 1.0  # [kg] mass of link2
        self.I2 = 1.0  # [kg*m^2] inertia around the center of mass of link 2
        self.l2 = 1.0  # [m] length of link2
        self.lc2 = 0.5  # [m] center of mass position of link2

        self.g = 9.8  # [kg*m/s^2] gravity
        self.dt = 0.02

        # limits
        self.max_link1_dtheta = 4 * np.pi
        self.max_link2_dtheta = 9 * np.pi

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        if batched:
            raise NotImplementedError
        x = x.flatten()
        u = u.flatten()
        x_and_u = np.append(x, u)
        (dtheta1, dtheta2, ddtheta1, ddtheta2, _) = self._dsdt(x_and_u)
        theta1_next = wrap(x[0] + dtheta1 * self.dt, -np.pi, np.pi)
        theta2_next = wrap(x[1] + dtheta2 * self.dt, -np.pi, np.pi)
        dtheta1_next = bound(x[2] + ddtheta1 * self.dt, -self.max_link1_dtheta, self.max_link1_dtheta)
        dtheta2_next = bound(x[3] + ddtheta2 * self.dt, -self.max_link2_dtheta, self.max_link2_dtheta)
        x_next = np.asarray([[theta1_next], [theta2_next], [dtheta1_next], [dtheta2_next]], dtype=np.float32)

        return x_next, {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if batched:
            raise NotImplementedError

        def non_wrapped_next_state(x, u, t):
            x = x.flatten()
            u = u.flatten()
            x_and_u = np.append(x, u)
            (dtheta1, dtheta2, ddtheta1, ddtheta2, _) = self._dsdt(x_and_u)
            # Remove wrapping to avoid unexpected gradient computation
            theta1_next = x[0] + dtheta1 * self.dt
            theta2_next = x[1] + dtheta2 * self.dt
            dtheta1_next = bound(x[2] + ddtheta1 * self.dt, -self.max_link1_dtheta, self.max_link1_dtheta)
            dtheta2_next = bound(x[3] + ddtheta2 * self.dt, -self.max_link2_dtheta, self.max_link2_dtheta)

            x_next = np.asarray([[theta1_next], [theta2_next], [dtheta1_next], [dtheta2_next]], dtype=np.float32)
            return x_next

        fx = non_wrapped_next_state(x, u, t).flatten()
        (x1, x2, x3, x4) = x.flatten()
        (fx1, fx2, fx3, fx4) = fx[0], fx[1], fx[2], fx[3]

        eps = 0.001
        dx1 = np.asarray([[x1 + eps], [x2], [x3], [x4]])
        fx_dx1 = non_wrapped_next_state(dx1, u, t).flatten()
        (fx1_dx1, fx2_dx1, fx3_dx1, fx4_dx1) = fx_dx1[0], fx_dx1[1], fx_dx1[2], fx_dx1[3]

        dx2 = np.asarray([[x1], [x2 + eps], [x3], [x4]])
        fx_dx2 = non_wrapped_next_state(dx2, u, t).flatten()
        (fx1_dx2, fx2_dx2, fx3_dx2, fx4_dx2) = fx_dx2[0], fx_dx2[1], fx_dx2[2], fx_dx2[3]

        dx3 = np.asarray([[x1], [x2], [x3 + eps], [x4]])
        fx_dx3 = non_wrapped_next_state(dx3, u, t).flatten()
        (fx1_dx3, fx2_dx3, fx3_dx3, fx4_dx3) = fx_dx3[0], fx_dx3[1], fx_dx3[2], fx_dx3[3]

        dx4 = np.asarray([[x1], [x2], [x3], [x4 + eps]])
        fx_dx4 = non_wrapped_next_state(dx4, u, t).flatten()
        (fx1_dx4, fx2_dx4, fx3_dx4, fx4_dx4) = fx_dx4[0], fx_dx4[1], fx_dx4[2], fx_dx4[3]

        du = u + eps
        fx_du = non_wrapped_next_state(x, du, t).flatten()
        (fx1_du, fx2_du, fx3_du, fx4_du) = fx_du[0], fx_du[1], fx_du[2], fx_du[3]

        Fx = np.asarray([[(fx1_dx1 - fx1) / eps, (fx1_dx2 - fx1) / eps, (fx1_dx3 - fx1) / eps, (fx1_dx4 - fx1) / eps],
                         [(fx2_dx1 - fx2) / eps, (fx2_dx2 - fx2) / eps, (fx2_dx3 - fx2) / eps, (fx2_dx4 - fx2) / eps],
                         [(fx3_dx1 - fx3) / eps, (fx3_dx2 - fx3) / eps, (fx3_dx3 - fx3) / eps, (fx3_dx4 - fx3) / eps],
                         [(fx4_dx1 - fx4) / eps, (fx4_dx2 - fx4) / eps, (fx4_dx3 - fx4) / eps, (fx4_dx4 - fx4) / eps]])
        Fu = np.asarray([[(fx1_du - fx1) / eps],
                         [(fx2_du - fx2) / eps],
                         [(fx3_du - fx3) / eps],
                         [(fx4_du - fx4) / eps]])

        return Fx, Fu

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dim(self) -> int:
        return 4

    def action_dim(self) -> int:
        return 1

    def _dsdt(self, x_and_u):
        x = x_and_u[:-1]
        u = x_and_u[-1]
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]
        d1 = (
            self.m1 * self.lc1**2
            + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * np.cos(theta2))
            + self.I1
            + self.I2
        )
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * np.cos(theta2)) + self.I2
        phi2 = self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -self.m2 * self.l1 * self.lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * self.m2 * self.l1 * self.lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(theta1 - np.pi / 2)
            + phi2
        )
        ddtheta2 = (u + d2 / d1 * phi1 - self.m2 * self.l1 * self.lc2 * dtheta1**2 *
                    np.sin(theta2) - phi2) / (self.m2 * self.lc2**2 + self.I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)


class AcrobotCostFunction(CostFunction):
    def __init__(self, T):
        super().__init__()
        weight_th1 = 500.0
        weight_th1dot = 100.0
        weight_th2 = 500.0
        weight_th2dot = 100.0
        self._weight_u = 0.1
        self._x_target = np.asarray([[np.pi], [0.0], [0.0], [0.0]], dtype=np.float32)
        self.T = T
        self.Q = np.zeros(shape=(4, 4))
        self.Q_final = np.asarray([[weight_th1, 0, 0, 0],
                                   [0, weight_th2, 0, 0],
                                   [0, 0, weight_th1dot, 0],
                                   [0, 0, 0, weight_th2dot]])

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        if batched:
            raise NotImplementedError
        # Stabilize around equilibrium point
        if x[0] < 0.0:
            x_target = -self._x_target
        else:
            x_target = self._x_target
        if final_state:
            # final state
            x_e = x - x_target
            state_cost = (x_e).T.dot(self.Q_final).dot(x_e)
            return state_cost
        else:
            # non final state
            # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
            # error vector
            state_cost = (x).T.dot(self.Q).dot(x)
            act_cost = self._weight_u * (u.flatten()**2)
            return state_cost + act_cost

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        # Stabilize around equilibrium point
        if x[0] < 0.0:
            x_target = -self._x_target
        else:
            x_target = self._x_target
        if final_state:
            # final state
            # error vector
            x_e = x - x_target
            # Return column vector
            Cx = 2 * self.Q_final.dot(x_e)
            return (Cx, None)
        else:
            # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
            # Return column vector
            Cx = 2 * self.Q.dot(x)
            Cu = 2 * self._weight_u * u
            return (Cx, Cu)

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        if final_state:
            # final state
            return 2.0 * self.Q_final, None, None, None
        else:
            x_dim = x.shape[0]
            u_dim = u.shape[0]
            Cux = np.zeros((u_dim, x_dim))
            Cxu = np.zeros((x_dim, u_dim))
            return 2.0 * self.Q, Cxu, Cux, 2.0 * np.eye(1) * self._weight_u


def compute_initial_trajectory(x0, dynamics, T, u):
    trajectory = []
    x = x0
    for t in range(T - 1):
        trajectory.append((x, u[t]))
        x, _ = dynamics.next_state(x, u[t], t)
    # append final state
    trajectory.append((x, None))
    return trajectory


def run_control(args):
    env = ContinuousAcrobot()

    dynamics = AcrobotDynamics()
    cost_function = AcrobotCostFunction(args.T)

    config = iLQRConfig(T_max=args.T, num_iterations=5)
    ilqr = iLQR(env, dynamics, cost_function, config)

    for _ in range(args.num_episodes):
        state = env.reset()
        state = np.reshape(state, (4, 1))
        done = False
        total_reward = 0
        initial_u = [np.zeros((dynamics.action_dim(), 1)) for t in range(args.T - 1)]

        while not done:
            # optimize
            initial_trajectory = compute_initial_trajectory(state, dynamics, args.T, initial_u)
            start = time.time()
            improved_trajectory, trajectory_info = ilqr.compute_trajectory(initial_trajectory)
            end = time.time()
            print(f'optimization time: {end - start} [s]')

            u = improved_trajectory[0][1]
            next_state, reward, done, info = env.step(u)
            total_reward += reward
            if args.render:
                env.render()
            theta1 = state[0] / np.pi * 180.0
            theta2 = state[1] / np.pi * 180.0
            dtheta1 = state[2] / np.pi * 180.0
            dtheta2 = state[3] / np.pi * 180.0
            print(f'control input torque: {u}')
            print(f'theta1: {theta1} [deg]')
            print(f'theta2: {theta2} [deg]')
            print(f'dtheta1: {dtheta1} [deg/s]')
            print(f'dtheta2: {dtheta2} [deg/s]')

            state = next_state
            state = np.reshape(state, (4, 1))
            initial_u = [x_u[1] for x_u in improved_trajectory[1:-1]]
            initial_u.append(np.zeros((dynamics.action_dim(), 1)))

        print(total_reward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=25)
    args = parser.parse_args()
    run_control(args)


if __name__ == '__main__':
    main()
