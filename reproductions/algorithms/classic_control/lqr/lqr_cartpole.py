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
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym import logger, spaces
from gym.envs.classic_control.cartpole import CartPoleEnv

from nnabla_rl.algorithms.lqr import LQR, LQRConfig
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


class ContinuousCartPole(CartPoleEnv):
    '''Continuous CartPole Environment
    '''

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(np.array([-5.0]), np.array([5.0]), dtype=np.float32)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}


class ContinuousCartPoleLinearDynamics(Dynamics):
    def __init__(self):
        super().__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self._A = np.zeros((4, 4))
        self._A[0, 1] = 1.
        self._A[1, 2] = - self.polemass_length / self.total_mass
        self._A[2, 3] = 1.
        denom = self.length * (4. / 3. - (self.masspole / self.total_mass))
        self._A[3, 2] = self.gravity / denom
        self._A = self._A * self.tau + np.eye(4)

        self._B = np.zeros((4, 1))
        self._B[1, 0] = 1. / self.total_mass
        self._B[3, 0] = (- 1. / self.total_mass) / denom
        self._B = self._B * self.tau

    def next_state(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        if batched:
            raise NotImplementedError
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        return self._A.dot(x) + self._B.dot(u), {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        if batched:
            raise NotImplementedError
        return self._A, self._B

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dim(self) -> int:
        return 4

    def action_dim(self) -> int:
        return 1


class QuadraticCostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        self._Q = np.eye(4) * 10
        self._R = np.eye(1) * 0.1
        self._F = np.zeros((4, 1))

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        if batched:
            raise NotImplementedError
        if final_state:
            return x.T.dot(self._Q).dot(x)
        else:
            # Assuming that target state is zero
            return x.T.dot(self._Q).dot(x) + 2.0*x.T.dot(self._F).dot(u) + u.T.dot(self._R).dot(u)

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        if final_state:
            return 2.0 * self._Q.dot(x), None
        else:
            return 2.0 * self._Q.dot(x) + 2.0 * self._F.dot(u), 2.0 * self._R.dot(u) + 2.0 * self._F.dot(x)

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if batched:
            raise NotImplementedError
        if final_state:
            return self._Q, None, None, None, None
        else:
            return self._Q, self._F, self._F, self._R


def compute_initial_trajectory(x0, dynamics, T, u):
    trajectory = []
    x = x0
    for t in range(T - 1):
        trajectory.append((x, u[t]))
        x, _ = dynamics.next_state(x, u[t], t)
    trajectory.append((x, None))
    return trajectory


def run_control(args):
    env = ContinuousCartPole()
    dynamics = ContinuousCartPoleLinearDynamics()
    cost_function = QuadraticCostFunction()

    config = LQRConfig(T_max=args.T)
    lqr = LQR(env, dynamics, cost_function, config)

    for _ in range(args.num_episodes):
        state = env.reset()
        state = np.reshape(state, (state.shape[0], 1))
        done = False
        total_reward = 0
        initial_u = [np.zeros((dynamics.action_dim(), 1)) for t in range(args.T - 1)]
        while not done:
            initial_trajectory = compute_initial_trajectory(state, dynamics, args.T, initial_u)
            start = time.time()
            improved_trajectory, trajectory_info = lqr.compute_trajectory(initial_trajectory)
            end = time.time()
            print(f'optimization time: {end - start} [s]')

            next_state, reward, done, info = env.step(improved_trajectory[0][1].flatten().astype(np.float32))
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
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()
    run_control(args)


if __name__ == '__main__':
    main()
