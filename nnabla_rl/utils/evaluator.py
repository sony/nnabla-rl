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

from typing import Callable, List, Union

import numpy as np

from nnabla_rl.logger import logger
from nnabla_rl.typing import Experience


class EpisodicEvaluator:
    def __init__(self, run_per_evaluation=10):
        self._num_episodes = run_per_evaluation

    def __call__(self, algorithm, env):
        returns = []
        for num in range(1, self._num_episodes + 1):
            reward_sum, *_ = run_one_episode(algorithm, env)
            returns.append(reward_sum)
            logger.info(
                "Finished evaluation run: #{} out of {}. Total reward: {}".format(num, self._num_episodes, reward_sum)
            )
        return returns


class TimestepEvaluator:
    def __init__(self, num_timesteps):
        self._num_timesteps = num_timesteps

    def __call__(self, algorithm, env):
        returns = []
        timesteps = 0

        def limit_checker(t):
            return t + timesteps > self._num_timesteps

        while True:
            reward_sum, episode_timesteps, *_ = run_one_episode(algorithm, env, timestep_limit=limit_checker)
            timesteps += episode_timesteps

            if timesteps > self._num_timesteps:
                break

            returns.append(reward_sum)
            logger.info(
                "Finished evaluation run: Time step #{} out of {}, Episode #{}. Total reward: {}".format(
                    timesteps, self._num_timesteps, len(returns), reward_sum
                )
            )
        if len(returns) == 0:
            # In case the time limit reaches on first episode, save the return received up to that time
            returns.append(reward_sum)
        return returns


class EpisodicSuccessEvaluator:
    def __init__(self, check_success: Callable[[List[Experience]], Union[bool, float]], run_per_evaluation=10):
        self._num_episodes = run_per_evaluation
        self._compute_success_func = check_success

    def __call__(self, algorithm, env):
        results = []
        for num in range(1, self._num_episodes + 1):
            _, _, experiences = run_one_episode(algorithm, env)
            success = self._compute_success_func(experiences)
            results.append(success)
            success_tag = "Success" if success else "Failed"
            logger.info("Finished evaluation run: #{} out of {}. {}".format(num, self._num_episodes, success_tag))

        return results


def run_one_episode(algorithm, env, timestep_limit=lambda t: False):
    experiences = []
    rewards = []
    timesteps = 0
    state = env.reset()
    extra_info = {"reward": 0}
    action = algorithm.compute_eval_action(state, begin_of_episode=True, extra_info=extra_info)
    while True:
        next_state, reward, done, info = env.step(action)
        non_terminal = 0.0 if done else 1.0
        experience = (state, action, reward, non_terminal, next_state, info)
        experiences.append(experience)

        rewards.append(reward)
        timesteps += 1
        if done or timestep_limit(timesteps):
            break
        else:
            state = next_state
            extra_info["reward"] = reward
            action = algorithm.compute_eval_action(state, begin_of_episode=False, extra_info=extra_info)
    return np.sum(rewards), timesteps, experiences
