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
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

from nnabla_rl.external.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv,
                                               NoopResetEnv, ScaledFloatFrame)

cv2.ocl.setUseOpenCL(False)


class CHWWarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        obs_shape = (1, self.height, self.width)  # 'chw' order
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=obs_shape, dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame.reshape(self.observation_space.low.shape)


class CHWFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        stack_axis = 0  # 'chw' order
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return CHWLazyFrames(list(self.frames))


class CHWLazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, normalize=True, frame_stack=True, fire_reset=False):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = CHWWarpFrame(env)
    if normalize:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = CHWFrameStack(env, 4)
    return env
