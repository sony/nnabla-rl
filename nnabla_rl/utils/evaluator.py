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

import numpy as np

from nnabla_rl.logger import logger


class EpisodicEvaluator():
    def __init__(self, run_per_evaluation=10):
        self._num_episodes = run_per_evaluation

    def __call__(self, algorithm, env):
        returns = []
        for num in range(1, self._num_episodes + 1):
            reward_sum, _ = run_one_episode(algorithm, env)
            returns.append(reward_sum)
            logger.info(
                'Finished evaluation run: #{} out of {}. Total reward: {}'
                .format(num, self._num_episodes, reward_sum))
        return returns


class TimestepEvaluator():
    def __init__(self, num_timesteps):
        self._num_timesteps = num_timesteps

    def __call__(self, algorithm, env):
        returns = []
        timesteps = 0

        def limit_checker(t):
            return t + timesteps > self._num_timesteps

        while True:
            reward_sum, episode_timesteps = run_one_episode(
                algorithm, env, timestep_limit=limit_checker)
            timesteps += episode_timesteps

            if timesteps > self._num_timesteps:
                break

            returns.append(reward_sum)
            logger.info(
                'Finished evaluation run: Time step #{} out of {}, Episode #{}. Total reward: {}'
                .format(timesteps, self._num_timesteps, len(returns), reward_sum))
        if len(returns) == 0:
            # In case the time limit reaches on first episode, save the return received up to that time
            returns.append(reward_sum)
        return returns


def run_one_episode(algorithm, env, timestep_limit=lambda t: False):
    rewards = []
    timesteps = 0
    state = env.reset()
    while True:
        action = algorithm.compute_eval_action(state)
        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)
        timesteps += 1
        if done or timestep_limit(timesteps):
            break
        else:
            state = next_state
    return np.sum(rewards), timesteps
