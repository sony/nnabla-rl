# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import numpy as np


class Float32ObservationEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(Float32ObservationEnv, self).__init__(env)
        self.dtype = np.float32
        self.observation_space.dtype = np.dtype(np.float32)

    def observation(self, observation):
        return self.dtype(observation)


class Float32RewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(Float32RewardEnv, self).__init__(env)
        self.dtype = np.float32

    def reward(self, reward):
        return self.dtype(reward)


class Float32ActionEnv(gym.ActionWrapper):
    def __init__(self, env):
        super(Float32ActionEnv, self).__init__(env)
        self.dtype = np.float32

    def action(self, action):
        return self.dtype(action)

    def reverse_action(self, action):
        return self.env.action_space.dtype(action)


class Int32ActionEnv(gym.ActionWrapper):
    def __init__(self, env):
        super(Int32ActionEnv, self).__init__(env)
        self.dtype = np.int32

    def action(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]
        return self.dtype(action)

    def reverse_action(self, action):
        return self.env.action_space.dtype(action)


class NumpyFloat32Env(gym.Wrapper):
    def __init__(self, env):
        env = Float32ObservationEnv(env)
        env = Float32RewardEnv(env)
        if isinstance(env.action_space, gym.spaces.Discrete):
            env = Int32ActionEnv(env)
        else:
            env = Float32ActionEnv(env)

        super(NumpyFloat32Env, self).__init__(env)


class ScreenRenderEnv(gym.Wrapper):
    def __init__(self, env):
        super(ScreenRenderEnv, self).__init__(env)
        self._env_name = env.unwrapped.spec.id

    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self):
        if 'Bullet' in self._env_name:
            self.env.render()
            state = self.env.reset()
        else:
            state = self.env.reset()
            self.env.render()
        return state
