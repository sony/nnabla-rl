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

import math
from typing import Optional, Tuple

import numpy as np
from gym.envs.box2d.lunar_lander import (FPS, LEG_DOWN, MAIN_ENGINE_POWER, SCALE, SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT,
                                         SIDE_ENGINE_POWER, VIEWPORT_H, VIEWPORT_W, LunarLander)
from gym.envs.mujoco.ant_v4 import AntEnv
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v4 import HopperEnv
from gym.envs.mujoco.humanoid_v4 import HumanoidEnv
from gym.envs.mujoco.walker2d_v4 import Walker2dEnv

from nnabla_rl.typing import Action, Info, Reward, State


class FactoredLunarLanderV2(LunarLander):
    prev_state: Optional[np.ndarray]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        super().__init__(
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        self.prev_state = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> State:
        state = super().reset(seed=seed, return_info=return_info, options=options)
        self.prev_state = np.array(state)
        return state  # type: ignore

    # https://github.com/openai/gym/blob/a8d4dd7b147fc5a3cec995293bb656967d0ab60f/gym/envs/box2d/lunar_lander.py#L443
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        assert self.lander is not None

        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # shaping rewards
        prev_state = np.zeros_like(state) if self.prev_state is None else self.prev_state
        reward_position = -100 * (np.sqrt(state[0] ** 2 + state[1] ** 2) -
                                  np.sqrt(prev_state[0] ** 2 + prev_state[1] ** 2))
        reward_velocity = -100 * (np.sqrt(state[2] ** 2 + state[3] ** 2) -
                                  np.sqrt(prev_state[2] ** 2 + prev_state[3] ** 2))
        reward_angle = -100 * (abs(state[4]) - abs(prev_state[4]))
        reward_left_leg = 10 * (state[6] - prev_state[6])
        reward_right_leg = 10 * (state[7] - prev_state[7])
        self.prev_state = np.array(state)

        # control conts
        reward_main_engine = -m_power * 0.30
        reward_side_engine = -s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward_failure = -100.0
        else:
            reward_failure = 0.0

        if not self.lander.awake:
            terminated = True
            reward_success = 100.0
        else:
            reward_success = 0.0

        if self.render_mode == "human":
            self.render()

        reward = [reward_position, reward_velocity, reward_angle, reward_left_leg, reward_right_leg,
                  reward_main_engine, reward_side_engine, reward_failure, reward_success]

        return np.array(state, dtype=np.float32), np.array(reward), terminated, {}


class FactoredAntV4(AntEnv):
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        observation, _, terminated, _, info = super().step(action)
        reward = [info["reward_forward"], info["reward_ctrl"], info["reward_survive"]]
        return observation, np.array(reward), terminated, info


class FactoredHopperV4(HopperEnv):
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        observation, _, terminated, _, info = super().step(action)

        forward_reward = self._forward_reward_weight * info["x_velocity"]
        healthy_reward = self.healthy_reward
        ctrl_cost = -self.control_cost(action)
        reward = np.array([forward_reward, healthy_reward, ctrl_cost])

        return observation, reward, terminated, info


class FactoredHalfCheetahV4(HalfCheetahEnv):
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        observation, _, terminated, _, info = super().step(action)
        reward = [info["reward_run"], info["reward_ctrl"]]
        return observation, np.array(reward), terminated, info


class FactoredWalker2dV4(Walker2dEnv):
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        observation, _, terminated, _, info = super().step(action)

        forward_reward = self._forward_reward_weight * info["x_velocity"]
        healthy_reward = self.healthy_reward
        ctrl_cost = -self.control_cost(action)
        reward = np.array([forward_reward, healthy_reward, ctrl_cost])

        return observation, reward, terminated, info


class FactoredHumanoidV4(HumanoidEnv):
    def step(self, action: Action) -> Tuple[State, Reward, bool, Info]:  # type: ignore
        observation, _, terminated, _, info = super().step(action)
        reward = [info["reward_linvel"], info["reward_quadctrl"], info["reward_alive"]]
        return observation, np.array(reward), terminated, info
