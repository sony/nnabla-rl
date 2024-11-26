# Copyright 2024 Sony Group Corporation.
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
import gymnasium

from nnabla_rl.environments.wrappers import Gymnasium2GymWrapper, NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.environments.wrappers.atari import CHWFrameStack, CHWWarpFrame
from nnabla_rl.external.atari_wrappers import ClipRewardEnv, NoopResetEnv, ScaledFloatFrame
from nnabla_rl.utils.reproductions import print_env_info


class EpisodicLifeOptionCriticEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        self._obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return self._obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            self._obs = self.env.reset(**kwargs)
        else:
            # NOTE: In option critic author env, they did not reset with noop action use previous obs instead.
            pass

        self.lives = self.env.unwrapped.ale.lives()
        return self._obs


class SkipOptionCriticEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame."""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def wrap_option_critic_deepmind(
    env,
    episode_life=True,
    clip_rewards=True,
):
    if episode_life:
        env = EpisodicLifeOptionCriticEnv(env)

    env = CHWWarpFrame(env)

    env = ScaledFloatFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = CHWFrameStack(env, 4)
    return env


def make_option_critic_atari(env_id, max_frames_per_episode=None, use_gymnasium=False):
    if use_gymnasium:
        env = gymnasium.make(env_id)
        env = Gymnasium2GymWrapper(env)
        # gymnasium env is not wrapped TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env.spec.kwargs["max_num_frames_per_episode"])
    else:
        env = gym.make(env_id)
    if max_frames_per_episode is not None:
        env = env.unwrapped
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_frames_per_episode)
    assert "NoFrameskip" in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = NoopResetEnv(env, noop_max=30)
    env = SkipOptionCriticEnv(env, skip=4)
    return env


def build_option_critic_atari_env(
    id_or_env,
    test=False,
    seed=None,
    render=False,
    print_info=True,
    max_frames_per_episode=None,
    use_gymnasium=False,
):
    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    elif isinstance(id_or_env, gymnasium.Env):
        env = Gymnasium2GymWrapper(id_or_env)
    else:
        env = make_option_critic_atari(
            id_or_env, max_frames_per_episode=max_frames_per_episode, use_gymnasium=use_gymnasium
        )
    if print_info:
        print_env_info(env)

    env = wrap_option_critic_deepmind(env, episode_life=not test, clip_rewards=not test)
    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.unwrapped.ale.setBool("color_averaging", False)
    env.seed(seed)
    return env
