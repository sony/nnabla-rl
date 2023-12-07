# Copyright 2023 Sony Group Corporation.
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

from typing import cast

import gym
import gym.spaces
import numpy as np
from gym.core import Env


class FlattenActionWrapper(gym.Wrapper):
    """Flatten the action_space and fix action order for goal env."""

    def __init__(self, env):
        super().__init__(env)
        original_action_space = env.action_space
        num_actions = original_action_space.spaces[0].n
        discrete_space = original_action_space.spaces[0]
        continuous_space = [gym.spaces.Box(original_action_space.spaces[1].spaces[i].low,
                                           original_action_space.spaces[1].spaces[i].high,
                                           dtype=np.float32) for i in range(0, num_actions)]
        self.action_space = gym.spaces.Tuple((discrete_space, *continuous_space))

    def step(self, action):
        action = (action[0], tuple(a for a in action[1:]))
        return super().step(action)


class ScaleStateWrapper(gym.ObservationWrapper):
    """Observation should be flatten in prior to merge."""

    def __init__(self, env: Env):
        super().__init__(env)
        self._original_observation_space = env.observation_space

        if self._is_box(env.observation_space):
            self.observation_space = gym.spaces.Box(low=-np.ones(shape=env.observation_space.shape),
                                                    high=np.ones(shape=env.observation_space.shape),
                                                    dtype=np.float32)
        elif self._is_tuple(env.observation_space):
            spaces = [gym.spaces.Box(low=-np.ones(shape=space.shape),
                                     high=np.ones(shape=space.shape),
                                     dtype=np.float32)
                      if self._is_box(space) else space for space in cast(gym.spaces.Tuple, env.observation_space)]
            self.observation_space = gym.spaces.Tuple(spaces)  # type: ignore
        else:
            raise NotImplementedError

    def observation(self, observation):
        if self._is_tuple(self.observation_space):
            return tuple(self._normalize(o, space) for (o, space) in zip(observation, self._original_observation_space))
        else:
            return self._normalize(observation, self._original_observation_space)

    def _normalize(self, observation, space):
        if not self._is_box(space):
            return observation
        range = space.high - space.low
        return 2.0 * (observation - space.low) / range - 1

    def _is_box(self, space):
        return isinstance(space, gym.spaces.Box)

    def _is_tuple(self, space):
        return isinstance(space, gym.spaces.Tuple)


class ScaleActionWrapper(gym.ActionWrapper):
    """Action should be flatten in prior to merge."""

    def __init__(self, env: Env):
        super().__init__(env)
        self._original_action_space = env.action_space

        if self._is_box(env.action_space):
            self.action_space = gym.spaces.Box(low=-np.ones(shape=env.action_space.shape),
                                               high=np.ones(shape=env.action_space.shape),
                                               dtype=np.float32)
        elif self._is_tuple(env.action_space):
            spaces = [gym.spaces.Box(low=-np.ones(shape=space.shape),
                                     high=np.ones(shape=space.shape),
                                     dtype=np.float32)
                      if self._is_box(space) else space for space in cast(gym.spaces.Tuple, env.action_space)]
            self.action_space = gym.spaces.Tuple(spaces)  # type: ignore
        else:
            raise NotImplementedError

    def action(self, action):
        if self._is_tuple(self.action_space):
            return tuple(self._unnormalize(a, space) for (a, space) in zip(action, self._original_action_space))
        else:
            return self._unnormalize(action, self._original_action_space)

    def reverse_action(self, action):
        raise NotImplementedError

    def _unnormalize(self, action, space):
        if not self._is_box(space):
            return action
        range = space.high - space.low
        return 0.5 * (action + 1) * range + space.low

    def _is_box(self, space):
        return isinstance(space, gym.spaces.Box)

    def _is_tuple(self, space):
        return isinstance(space, gym.spaces.Tuple)


class MergeBoxActionWrapper(gym.ActionWrapper):
    """Action should be flatten in prior to merge."""

    def __init__(self, env: Env):
        super().__init__(env)
        original_action_space = cast(gym.spaces.Tuple, env.action_space)
        self._original_action_size = [np.prod(space.shape) for space in original_action_space[1:]]
        self._original_action_shape = [space.shape for space in original_action_space[1:]]
        self._original_action_low = [space.low for space in original_action_space[1:]]
        self._original_action_range = [space.high - space.low for space in original_action_space[1:]]
        action_size = max(self._original_action_size)
        d_action_space = original_action_space[0]
        c_action_space = gym.spaces.Box(low=-np.ones(shape=(action_size, )),
                                        high=np.ones(shape=(action_size, )),
                                        dtype=np.float32)
        self.action_space = gym.spaces.Tuple((d_action_space, c_action_space))  # type: ignore

    def action(self, action):
        (d_action, c_action) = action
        expanded_actions = []
        for i, size in enumerate(self._original_action_size):
            shape = self._original_action_shape[i]
            a = c_action[:size]
            a = np.reshape(a, newshape=shape)

            low = self._original_action_low[i]
            range = self._original_action_range[i]
            a = (a + 1.0) / 2.0 * range + low if i == d_action else np.zeros(shape=shape)
            expanded_actions.append(a)
        return (d_action, *expanded_actions)


class EmbedActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        original_action_space = cast(gym.spaces.Tuple, env.action_space)
        self.d_action_dim = original_action_space[0].n
        self.c_action_dim = np.prod(original_action_space[1].shape)
        self.action_space = self._create_action_space(env)
        self.embed_map = np.random.normal(size=(self.d_action_dim, self.d_action_dim))

    def action(self, action):
        d_action = self._decode(action[:self.d_action_dim])
        c_action = action[self.d_action_dim:]
        return (d_action, c_action)

    def reverse_action(self, action):
        raise NotImplementedError

    def _decode(self, action):
        return np.argmax((self.embed_map - action) ** 2, axis=0)

    def _create_action_space(self, env):
        (d_space, c_space) = env.action_space
        return gym.spaces.Box(-1, 1, shape=(d_space.n + np.prod(c_space.shape)))


class RemoveStepWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Tuple):  # type: ignore
            raise ValueError('observation space is not a tuple!')
        self.observation_space = cast(gym.spaces.Tuple, env.observation_space)[0]

    def observation(self, observation):
        (state, _) = observation
        return state
