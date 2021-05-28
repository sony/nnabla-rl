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
from gym.envs.registration import EnvSpec


class AbstractDummyEnv(gym.Env):
    def __init__(self, max_episode_steps):
        self.spec = EnvSpec(id='dummy-v0', max_episode_steps=max_episode_steps)

    def reset(self):
        return self.observation_space.sample()

    def step(self, a):
        next_state = self.observation_space.sample()
        reward = np.random.randn()
        done = False
        info = {}
        return next_state, reward, done, info


class DummyContinuous(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyContinuous, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))


class DummyDiscrete(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyDiscrete, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))


class DummyDiscreteImg(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyDiscreteImg, self).__init__(
            max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 84, 84))
        self.action_space = gym.spaces.Discrete(4)


class DummyAtariEnv(AbstractDummyEnv):
    class DummyALE(object):
        def __init__(self):
            self._lives = 100

        def lives(self):
            self._lives -= 1
            if self._lives < 0:
                self._lives = 100
            return self._lives

    np_random = np.random

    def __init__(self, done_at_random=True, max_episode_length=None):
        super(DummyAtariEnv, self).__init__(
            max_episode_steps=max_episode_length)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.ale = DummyAtariEnv.DummyALE()
        self._done_at_random = done_at_random
        self._max_episode_length = max_episode_length
        self._episode_length = None

    def step(self, action):
        assert self._episode_length is not None
        observation = self.observation_space.sample()
        self._episode_length += 1
        if self._done_at_random:
            done = (np.random.randint(10) == 0)
        else:
            done = False
        if self._max_episode_length is not None:
            done = (self._max_episode_length <= self._episode_length) or done
        return observation, 1.0, done, {'needs_reset': False}

    def reset(self):
        self._episode_length = 0
        return self.observation_space.sample()

    def get_action_meanings(self):
        return ['NOOP', 'FIRE', 'LEFT', 'RIGHT']


class DummyMujocoEnv(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyMujocoEnv, self).__init__(max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))

    def get_dataset(self):
        dataset = {}
        datasize = 100
        dataset['observations'] = np.stack([self.observation_space.sample() for _ in range(datasize)], axis=0)
        dataset['actions'] = np.stack([self.action_space.sample() for _ in range(datasize)], axis=0)
        dataset['rewards'] = np.random.randn(datasize, 1)
        dataset['terminals'] = np.zeros((datasize, 1))
        return dataset
