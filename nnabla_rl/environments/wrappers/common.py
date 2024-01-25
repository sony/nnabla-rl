# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

import cv2
import gym
import numpy as np
from gym import spaces
from packaging.version import parse

from nnabla_rl.logger import logger


class Float32ObservationEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(Float32ObservationEnv, self).__init__(env)
        self.dtype = np.float32
        if isinstance(env.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple(
                [self._create_observation_space(observation_space)
                 for observation_space in env.observation_space]
            )
        else:
            self.observation_space = self._create_observation_space(env.observation_space)

    def _create_observation_space(self, observation_space):
        if isinstance(observation_space, spaces.Box):
            return spaces.Box(
                low=observation_space.low,
                high=observation_space.high,
                shape=observation_space.shape,
                dtype=self.dtype
            )
        elif isinstance(observation_space, spaces.Discrete):
            return spaces.Discrete(n=observation_space.n)
        else:
            raise NotImplementedError

    def observation(self, observation):
        if isinstance(observation, tuple):
            return tuple(map(self.dtype, observation))
        else:
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
        self.continuous_dtype = np.float32
        self.discrete_dtype = np.int32
        if isinstance(env.action_space, spaces.Tuple):
            self.action_space = spaces.Tuple(
                [self._create_action_space(action_space) for action_space in env.action_space]
            )
        else:
            self.action_space = self._create_action_space(env.action_space)

    def action(self, action):
        def _action(action, action_space):
            if isinstance(action_space, spaces.Discrete):
                if isinstance(action, np.ndarray):
                    action = action[0]
                return self.discrete_dtype(action)
            else:
                return self.continuous_dtype(action)

        if isinstance(action, tuple):
            return tuple(_action(a, action_space) for a, action_space in zip(action, self.env.action_space))
        else:
            return self.continuous_dtype(action)

    def reverse_action(self, action):
        if isinstance(action, tuple):
            return tuple(action_space.dtype(action) for action_space in self.env.action_space)
        else:
            return self.env.action_space.dtype(action)

    def _create_action_space(self, action_space):
        if isinstance(action_space, spaces.Box):
            return spaces.Box(
                low=action_space.low,
                high=action_space.high,
                shape=action_space.shape,
                dtype=self.continuous_dtype
            )
        elif isinstance(action_space, spaces.Discrete):
            return spaces.Discrete(n=action_space.n)
        else:
            raise NotImplementedError


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
        self._installed_gym_version = parse(gym.__version__)
        self._gym_version25 = parse('0.25.0')
        self._env_name = "Unknown" if env.unwrapped.spec is None else env.unwrapped.spec.id

    def step(self, action):
        results = self.env.step(action)
        self._render_env()
        return results

    def reset(self):
        if 'Bullet' in self._env_name:
            self._render_env()
            state = self.env.reset()
        else:
            state = self.env.reset()
            self._render_env()
        return state

    def _render_env(self):
        if self._gym_version25 <= self._installed_gym_version:
            # 0.25.0 <= gym version
            rgb_array = self.env.render(mode='rgb_array')
            cv2.imshow(f'{self._env_name}', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        else:
            # old gym version
            self.env.render()


class PrintEpisodeResultEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._episode_rewards = []
        self._episode_num = 0

    def step(self, action):
        s_next, reward, done, info = self.env.step(action)
        self._episode_rewards.append(reward)
        if done:
            self._episode_num += 1
            episode_steps = len(self._episode_rewards)
            episode_return = np.sum(self._episode_rewards)
            logger.info(f'Episode #{self._episode_num} finished.')
            logger.info(f'Episode steps: {episode_steps}. Total return: {episode_return}.')
            self._episode_rewards.clear()
        return s_next, reward, done, info

    def reset(self):
        self._episode_rewards.clear()
        return self.env.reset()


class TimestepAsStateEnv(gym.Wrapper):
    """Timestep as state environment wrapper.

    This wrapper adds the timestep to original state. The concatenated
    state provides in TupleState type.
    """

    def __init__(self, env):
        super(TimestepAsStateEnv, self).__init__(env)
        self._timestep = 0
        obs_space = self.observation_space
        timestep_obs_space = spaces.Box(low=0., high=np.inf, shape=(1, ), dtype=np.float32)
        self.observation_space = spaces.Tuple([obs_space, timestep_obs_space])

    def reset(self):
        observation = self.env.reset()
        self._timestep = 0
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._timestep += 1
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return (observation, np.ones(1, dtype=np.int32) * self._timestep)


class HWCToCHWEnv(gym.ObservationWrapper):
    """HWC to CHW env wrapper.

    This wrapper changes the order of the image, from (height, width,
    channel) to (channel, height, width)
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        height, width, channel = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(channel, height, width), dtype=np.uint8)

    def observation(self, obs):
        return np.transpose(obs, [2, 0, 1])


class FlattenNestedTupleStateWrapper(gym.ObservationWrapper):
    """Flatten a nested tuple state observation wrapper.

    This wapper flattens a state.
    For example, if the original env_info.observation_shape is


    ```
    Tuple(Tuple(Box(-inf, inf, (2,), float32), Box(-inf, inf, (6,), float32)),
          Tuple(Box(-inf, inf, (1,), float32), Box(-inf, inf, (3,), float32)))
    ```,

    then the wrapped observation_shape is

    ```
    Tuple(Box(-inf, inf, (2,), float32),
          Box(-inf, inf, (6,), float32),
          Box(-inf, inf, (1,), float32),
          Box(-inf, inf, (3,), float32))
    ```.
    """

    def __init__(self, env):
        super().__init__(env)
        original_observation_space = env.observation_space
        assert isinstance(original_observation_space, gym.spaces.Tuple)
        self.observation_space = self._flatten_observation_space(original_observation_space)

    def _flatten_observation_space(self, observation_space):
        flattened_obs = []
        for space in observation_space:
            if isinstance(space, gym.spaces.Tuple):
                space = self._flatten_tuple_space(space)
                flattened_obs.extend(space)
            else:
                flattened_obs.append(space)
        return gym.spaces.Tuple(flattened_obs)

    def _flatten_tuple_space(self, tuple_space):
        flattened = []
        for item in tuple_space:
            if isinstance(item, gym.spaces.Tuple):
                flattened.extend(self._flatten_tuple_space(item))
            else:
                flattened.append(item)
        return flattened

    def observation(self, observation):
        flattened_obs = []
        for obs in observation:
            if isinstance(obs, tuple):
                obs = self._flatten_nested_array(obs)
                flattened_obs.extend(obs)
            else:
                flattened_obs.append(obs)
        return tuple(flattened_obs)

    def _flatten_nested_array(self, arr):
        flattened = []
        for item in arr:
            if isinstance(item, tuple):
                flattened.extend(self._flatten_nested_array(item))
            else:
                flattened.append(item)
        return flattened
