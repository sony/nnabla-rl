import gym

import numpy as np

from nnabla import random

from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv, make_atari, wrap_deepmind
from nnabla_rl.logger import logger
from nnabla_rl.replay_buffer import ReplayBuffer


def set_global_seed(seed, env=None):
    np.random.seed(seed=seed)
    random.prng = np.random.RandomState(seed=seed)
    if env is not None:
        env.seed(seed)


def build_atari_env(id_or_env, test=False, seed=None, render=False):
    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = make_atari(id_or_env)
    print_env_info(env)

    env = wrap_deepmind(env,
                        episode_life=not test,
                        clip_rewards=not test)
    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


def build_mujoco_env(id_or_env, test=False, seed=None, render=False):
    try:
        # Add pybullet env
        import pybullet_envs  # noqa
    except ModuleNotFoundError:
        # Do nothing if pybullet is not installed
        pass
    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = gym.make(id_or_env)
    print_env_info(env)

    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


def d4rl_dataset_to_buffer(dataset, max_buffer_size=1000000):
    buffer_size = min(dataset['observations'].shape[0], max_buffer_size)
    states = dataset['observations'][:buffer_size]
    actions = dataset['actions'][:buffer_size]
    rewards = dataset['rewards'][:buffer_size].reshape(buffer_size, 1)
    non_terminals = 1.0 - \
        dataset['terminals'][:buffer_size].reshape(buffer_size, 1)
    next_states = np.concatenate(
        [states[1:buffer_size, :], np.zeros(shape=states[0].shape)[np.newaxis, :]], axis=0)
    assert len(states) == buffer_size
    assert len(states) == len(actions)
    assert len(states) == len(rewards)
    assert len(states) == len(non_terminals)
    assert len(next_states) == len(next_states)
    buffer = ReplayBuffer(capacity=max_buffer_size)
    buffer.append_all(list(zip(states, actions, rewards,
                               non_terminals, next_states)))
    return buffer


def print_env_info(env):
    if env.unwrapped.spec is not None:
        env_name = env.unwrapped.spec.id
    else:
        env_name = 'Unknown'

    if isinstance(env.observation_space, gym.spaces.Discrete):
        state_dim = env.observation_space.n
        state_high = 'N/A'
        state_low = 'N/A'
    elif isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape
        state_high = env.observation_space.high
        state_low = env.observation_space.low

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        action_high = 'N/A'
        action_low = 'N/A'
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape
        action_high = env.action_space.high
        action_low = env.action_space.low

    info = f'''env: {env_name},
               space_dim/classes: {state_dim},
               space_high: {state_high},
               space_low: {state_low},
               action_dim/classes: {action_dim},
               action_high: {action_high},
               action_low: {action_low},
               max_episode_steps: {env.spec.max_episode_steps}'''
    logger.info(info)
