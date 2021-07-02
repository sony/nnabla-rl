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

import random as py_random

import gym
import numpy as np

import nnabla as nn
import nnabla_rl as rl
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv, make_atari, wrap_deepmind
from nnabla_rl.logger import logger


def set_global_seed(seed: int):
    np.random.seed(seed=seed)
    py_random.seed(seed)
    nn.seed(seed)
    rl.seed(seed)


def build_atari_env(id_or_env, test=False, seed=None, render=False, print_info=True, max_frames_per_episode=None):
    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = make_atari(id_or_env, max_frames_per_episode=max_frames_per_episode)
    if print_info:
        print_env_info(env)

    env = wrap_deepmind(env, episode_life=not test, clip_rewards=not test)
    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


def build_mujoco_env(id_or_env, test=False, seed=None, render=False, print_info=True):
    try:
        # Add pybullet env
        import pybullet_envs  # noqa
    except ModuleNotFoundError:
        # Ignore if pybullet is not installed
        pass
    try:
        # Add d4rl env
        import d4rl  # noqa
    except ModuleNotFoundError:
        # Ignore if d4rl is not installed
        pass

    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = gym.make(id_or_env)

    if print_info:
        print_env_info(env)

    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


def d4rl_dataset_to_experiences(dataset, size=1000000):
    size = min(dataset['observations'].shape[0], size)
    states = dataset['observations'][:size]
    actions = dataset['actions'][:size]
    rewards = dataset['rewards'][:size].reshape(size, 1)
    non_terminals = 1.0 - dataset['terminals'][:size].reshape(size, 1)
    next_states = np.concatenate([states[1:size, :], np.zeros(shape=states[0].shape)[np.newaxis, :]], axis=0)
    infos = [{} for _ in range(size)]
    assert len(states) == size
    assert len(states) == len(actions)
    assert len(states) == len(rewards)
    assert len(states) == len(non_terminals)
    assert len(states) == len(next_states)
    assert len(states) == len(infos)
    return list(zip(states, actions, rewards, non_terminals, next_states, infos))


def print_env_info(env):
    if env.unwrapped.spec is not None:
        env_name = env.unwrapped.spec.id
    else:
        env_name = 'Unknown'

    env_info = EnvironmentInfo.from_env(env)

    info = f'''env: {env_name},
               state_dim: {env_info.state_dim},
               state_shape: {env_info.state_shape},
               state_high: {env_info.state_high},
               state_low: {env_info.state_low},
               action_dim: {env_info.action_dim},
               action_shape: {env_info.action_shape},
               action_high: {env_info.action_high},
               action_low: {env_info.action_low},
               max_episode_steps: {env.spec.max_episode_steps}'''
    logger.info(info)
