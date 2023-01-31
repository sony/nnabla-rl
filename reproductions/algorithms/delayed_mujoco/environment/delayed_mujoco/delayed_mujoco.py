# Copyright 2022,2023 Sony Group Corporation.
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

import gym
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv


class DelayedEnv(gym.Wrapper):
    """Delayed Env Accumulated reward is returned every D timesteps."""

    def __init__(self, env, delay):
        super().__init__(env)
        self._delay = delay
        self._delayed_reward = 0.0
        self._count = 0

    def reset(self):
        self._delayed_reward = 0.0
        self._count = 0
        return super().reset()

    def step(self, a):
        ob, reward, done, info = super().step(a)
        self._delayed_reward += reward
        self._count += 1
        if (self._count == self._delay) or done:
            reward = self._delayed_reward
            self._delayed_reward = 0.0
            self._count = 0
        else:
            reward = 0.0
        return (ob, reward, done, info)


class DelayedAntEnv(DelayedEnv):
    def __init__(self):
        super().__init__(env=AntEnv(), delay=20)


class DelayedHalfCheetahEnv(DelayedEnv):
    def __init__(self):
        super().__init__(env=HalfCheetahEnv(), delay=20)


class DelayedHopperEnv(DelayedEnv):
    def __init__(self):
        super().__init__(env=HopperEnv(), delay=20)


class DelayedWalker2dEnv(DelayedEnv):
    def __init__(self):
        super().__init__(env=Walker2dEnv(), delay=20)
