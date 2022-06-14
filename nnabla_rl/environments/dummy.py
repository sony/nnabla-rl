# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

from typing import TYPE_CHECKING, cast

import gym
import numpy as np
from gym.envs.registration import EnvSpec

if TYPE_CHECKING:
    from gym.utils.seeding import RandomNumberGenerator

import nnabla_rl
from nnabla_rl.external.goal_env import GoalEnv


class AbstractDummyEnv(gym.Env):
    def __init__(self, max_episode_steps):
        self.spec = EnvSpec(id='dummy-v0', max_episode_steps=max_episode_steps)
        self._episode_steps = 0

    def reset(self):
        self._episode_steps = 0
        return self.observation_space.sample()

    def step(self, a):
        next_state = self.observation_space.sample()
        reward = np.random.randn()
        done = False if self.spec.max_episode_steps is None else bool(self._episode_steps < self.spec.max_episode_steps)
        info = {'rnn_states': {'dummy_scope': {'dummy_state1': 1, 'dummy_state2': 2}}}
        self._episode_steps += 1
        return next_state, reward, done, info


class DummyContinuous(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None, observation_shape=(5, ), action_shape=(5, )):
        super(DummyContinuous, self).__init__(
            max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=observation_shape)
        self.action_space = gym.spaces.Box(low=-1.0, high=5.0, shape=action_shape)


class DummyDiscrete(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyDiscrete, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(5)


class DummyTupleContinuous(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyTupleContinuous, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=0.0, high=1.0, shape=(4, )),
                                                   gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))))


class DummyTupleDiscrete(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyTupleDiscrete, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(5)))


class DummyTupleMixed(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyTupleMixed, self).__init__(
            max_episode_steps=max_episode_steps)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, ))
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(4),
                                                   gym.spaces.Box(low=0.0, high=1.0, shape=(5, ))))


class DummyDiscreteImg(AbstractDummyEnv):
    def __init__(self, max_episode_steps=None):
        super(DummyDiscreteImg, self).__init__(
            max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 84, 84))
        self.action_space = gym.spaces.Discrete(4)


class DummyContinuousImg(AbstractDummyEnv):
    def __init__(self, image_shape=(3, 64, 64), max_episode_steps=None):
        super(DummyContinuousImg, self).__init__(
            max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=image_shape)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))


class DummyAtariEnv(AbstractDummyEnv):
    class DummyALE(object):
        def __init__(self):
            self._lives = 100

        def lives(self):
            self._lives -= 1
            if self._lives < 0:
                self._lives = 100
            return self._lives

    # seeding.np_random outputs np_random and seed
    np_random = cast("RandomNumberGenerator", nnabla_rl.random.drng)

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
            done = bool(self.np_random.integers(10) == 0)
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


class DummyContinuousActionGoalEnv(GoalEnv):
    def __init__(self, max_episode_steps=10):
        self.spec = EnvSpec(id='dummy-continuou-action-goal-v0', max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Dict({'observation': gym.spaces.Box(low=0.0, high=1.0, shape=(5, )),
                                                  'achieved_goal': gym.spaces.Box(low=0.0, high=1.0, shape=(2, )),
                                                  'desired_goal': gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))})
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))
        self._max_episode_length = max_episode_steps
        self._episode_length = 0
        self._desired_goal = None

    def reset(self):
        super(DummyContinuousActionGoalEnv, self).reset()
        self._episode_length = 0
        state = self.observation_space.sample()
        self._desired_goal = state['desired_goal']
        return state

    def step(self, a):
        next_state = self.observation_space.sample()
        next_state['desired_goal'] = self._desired_goal
        reward = self.compute_reward(next_state['achieved_goal'], next_state['desired_goal'], {})
        self._episode_length += 1
        info = {'is_success': reward}
        if self._episode_length >= self._max_episode_length:
            done = True
        else:
            done = False
        return next_state, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.linalg.norm(achieved_goal - desired_goal) < 0.1:
            return 1
        else:
            return 0


class DummyDiscreteActionGoalEnv(GoalEnv):
    def __init__(self, max_episode_steps=10):
        self.spec = EnvSpec(id='dummy-discrete-action-goal-v0', max_episode_steps=max_episode_steps)
        self.observation_space = gym.spaces.Dict({'observation': gym.spaces.Box(low=0.0, high=1.0, shape=(5, )),
                                                  'achieved_goal': gym.spaces.Box(low=0.0, high=1.0, shape=(2, )),
                                                  'desired_goal': gym.spaces.Box(low=0.0, high=1.0, shape=(2, ))})
        self.action_space = gym.spaces.Discrete(n=3)
        self._max_episode_length = max_episode_steps
        self._episode_length = 0
        self._desired_goal = None

    def reset(self):
        super(DummyDiscreteActionGoalEnv, self).reset()
        self._episode_length = 0
        state = self.observation_space.sample()
        self._desired_goal = state['desired_goal']
        return state

    def step(self, a):
        next_state = self.observation_space.sample()
        next_state['desired_goal'] = self._desired_goal
        reward = self.compute_reward(next_state['achieved_goal'], next_state['desired_goal'], {})
        self._episode_length += 1
        info = {'is_success': reward}
        if self._episode_length >= self._max_episode_length:
            done = True
        else:
            done = False
        return next_state, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if np.linalg.norm(achieved_goal - desired_goal) < 0.1:
            return 1
        else:
            return 0
