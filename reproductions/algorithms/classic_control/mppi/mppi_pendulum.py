# Copyright 2022,2023,2024 Sony Group Corporation.
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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

import nnabla_rl.hooks as H
from nnabla_rl.algorithms.mppi import MPPI, MPPIConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environments.wrappers.common import ScreenRenderEnv
from nnabla_rl.hook import Hook
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics
from nnabla_rl.utils.data import unzip


class AnglePendulum(PendulumEnv):
    """Pendulum environment where the state is angle and angle speed."""

    def __init__(self, g=10):
        super().__init__(g)
        high = np.array([1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self._steps = 0
        self._max_steps = 1000

    def reset(self):
        initial_state = super().reset()
        self._steps = 0
        return initial_state

    def step(self, u):
        (next_state, reward, done, info) = super().step(u)
        self._steps += 1
        done = self._max_steps < self._steps or done
        return next_state, reward, done, info

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([angle_normalize(theta), thetadot], dtype=np.float32)


class PendulumKnownDynamics(Dynamics):
    def __init__(self):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.length = 1.0

    def next_state(
        self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        if batched:
            th, thdot = np.split(x, x.shape[-1], axis=-1)
            u = u
        else:
            th = x.flatten()[0]  # th := theta
            thdot = x.flatten()[1]
            u = u.flatten()[0]
        u = np.clip(u, -self.max_torque, self.max_torque)
        newthdot = thdot + (3 * self.g / (2 * self.length) * np.sin(th) + 3.0 / (self.m * self.length**2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        newth = angle_normalize(newth)
        if batched:
            return np.concatenate((newth, newthdot), axis=-1), {}
        else:
            return np.array([[newth], [newthdot]], dtype=np.float32), {}

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def hessian(
        self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dim(self) -> int:
        return 2

    def action_dim(self) -> int:
        return 1

    def support_batch(self) -> bool:
        return True


class PendulumCostFunction(CostFunction):
    def __init__(self):
        super().__init__()
        self._weight_th = 2.0
        self._weight_thdot = 0.5

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        # x.shape = (state_dim, 1) and u.shape = (input_dim, 1)
        if batched:
            th, thdot = np.split(x, x.shape[-1], axis=-1)
        else:
            th = x.flatten()[0]  # th := theta
            thdot = x.flatten()[1]
        state_cost = self._weight_th * self._angle_normalize(th) ** 2 + self._weight_thdot * thdot**2
        return state_cost

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError

    def support_batch(self) -> bool:
        return True

    def _angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


class MPPIEvaluationHook(Hook):
    def __init__(self, env, action_dim, T, timing=1):
        super().__init__(timing)
        self._env = env
        self._dummy_states = [None] * T
        self._control_inputs = np.zeros(shape=(T, action_dim))

    def on_hook_called(self, algorithm):
        x0 = self._env.reset()
        self._dummy_states[0] = x0
        self._control_inputs = np.zeros(shape=self._control_inputs.shape)
        done = False
        total_reward = 0
        while not done:
            trajectory = list(zip(self._dummy_states, self._control_inputs))
            trajectory = algorithm.compute_trajectory(trajectory)
            self._dummy_states, self._control_inputs = unzip(trajectory)

            u = self._control_inputs[0]
            x_next, reward, done, _ = self._env.step(u)

            self._dummy_states = list(self._dummy_states)
            self._control_inputs = np.asarray(self._control_inputs)
            self._dummy_states[0] = x_next
            self._control_inputs[0:-1] = self._control_inputs[1:]
            total_reward += reward
        print(f"total reward: {total_reward}")


def compute_initial_trajectory(x0, dynamics, T, u):
    trajectory = []
    x = x0
    for t in range(T - 1):
        trajectory.append((x, u[t]))
        x, _ = dynamics.next_state(x, u[t], t)
    trajectory.append((x, None))
    return trajectory


def run_control(args):
    train_env = AnglePendulum()
    eval_env = AnglePendulum()
    cost_function = PendulumCostFunction()

    if args.render:
        train_env = ScreenRenderEnv(train_env)

    if args.render:
        eval_env = ScreenRenderEnv(eval_env)

    env_info = EnvironmentInfo.from_env(eval_env)
    iteration_num_hook = H.IterationNumHook(timing=1)
    iteration_state_hook = H.IterationStateHook(timing=1)
    hooks = [iteration_num_hook, iteration_state_hook]

    covariance = np.eye(N=env_info.action_dim) * 0.5
    known_dynamics = PendulumKnownDynamics()
    config = MPPIConfig(
        gpu_id=args.gpu,
        T=args.T,
        covariance=covariance,
        K=1000,
        use_known_dynamics=args.use_known_dynamics,
        training_iterations=1000,
        dt=known_dynamics.dt,
    )

    def normalize_state(x):
        th, thdot = np.split(x, x.shape[-1], axis=-1)
        th = angle_normalize(th)
        return np.concatenate((th, thdot), axis=-1)

    if args.use_known_dynamics:
        mppi = MPPI(train_env, cost_function, known_dynamics, state_normalizer=normalize_state, config=config)
    else:
        mppi = MPPI(train_env, cost_function, None, state_normalizer=normalize_state, config=config)
    mppi.set_hooks(hooks)

    mppi.train_online(train_env, total_iterations=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--use-known-dynamics", action="store_true")
    args = parser.parse_args()
    run_control(args)


if __name__ == "__main__":
    main()
