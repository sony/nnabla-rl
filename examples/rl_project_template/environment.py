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
import gym
from gym.envs.registration import EnvSpec


class TemplateEnv(gym.Env):
    def __init__(self, max_episode_steps=100):
        # max_episode_steps is the maximum possible steps that the rl agent interacts with this environment.
        # You can set this value to None if the is no limits.
        # The first argument is the name of this environment used when registering this environment to gym.
        self.spec = EnvSpec('template-v0', max_episode_steps=max_episode_steps)
        self._episode_steps = 0

        # Use gym's spaces to define the shapes and ranges of states and actions.
        # observation_space: definition of states's shape and its ranges
        # action_space: definition of actions's shape and its ranges
        observation_shape = (10, )  # Example 10 dimensional state with range of [0.0, 1.0] each.
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=observation_shape)
        action_shape = (1, )  # 1 dimensional continuous action with range of [0.0, 1.0].
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=action_shape)

    def reset(self):
        # Reset the entire environment and return the initial state after the reset.
        # In this example, we just reset the steps to 0 and return a random state as initial state.
        self._episode_steps = 0

        return self.observation_space.sample()

    def step(self, action):
        # step is the core of the environment class.
        # You will need to compute the next_state according to the action received and
        # the reward to be received.
        # You will also need to set and return a done flag if the next_state is the end of the episode.

        # Increment episode steps
        self._episode_steps += 1

        next_state = self._compute_next_state(action)
        reward = self._compute_reward(next_state, action)

        # Here we set done flag to false if current episode steps exceeds
        # the max episode steps defined in this environment.
        done = False if self.spec.max_episode_steps is None else (self.spec.max_episode_steps <= self._episode_steps)

        # info is a convenient dictionary that you can fill
        # any additional information to return back to the RL algorithm.
        # If you have no extra information, return a empty dictionary.
        info = {}
        return next_state, reward, done, info

    def _compute_next_state(self, action):
        # In this template, we just return a randomly sampled state.
        # But in real application, you should compute the next_state according to the given action.
        return self.observation_space.sample()

    def _compute_reward(self, state, action):
        # In this template, we implemented a easy to understand reward function.
        # But in real an application, the design of this reward function is extremely important.
        if self._episode_steps < self.spec.max_episode_steps:
            return 0
        else:
            return 1
