# Copyright 2023,2024 Sony Group Corporation.
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

# We referred original author's code for this additional modification to Goal-v0 environment.
# See the supplemental material of the HyAR's paper.

import gym
import gym.spaces
import numpy as np
from gym_goal.envs.goal_env import GOAL_WIDTH, PITCH_LENGTH


class ExtendedGoalEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_observation_space = env.observation_space.spaces[0]
        extended_observation_shape = (17,)

        low = np.zeros(extended_observation_shape)
        low[:14] = original_observation_space.low
        low[14] = -1.0
        low[15] = -1.0
        low[16] = -GOAL_WIDTH / 2

        high = np.ones(extended_observation_shape)
        high[:14] = original_observation_space.high
        high[14] = 1.0
        high[15] = 1.0
        high[16] = GOAL_WIDTH

        max_steps = 200
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Box(low=low, high=high, dtype=np.float32), gym.spaces.Discrete(max_steps))
        )

    def observation(self, obs):
        state, steps = obs
        ball_feature = self._ball_feature(state)
        gk_feature = self._gk_feature(state)
        state = np.concatenate((state, ball_feature, gk_feature))
        return (state, steps)

    def _gk_feature(self, state):
        (ball_x, ball_y) = self._ball_position(state)
        (gk_x, gk_y) = self._gk_position(state)
        if gk_x == ball_x:
            feature = -GOAL_WIDTH / 2 if gk_y < ball_y else GOAL_WIDTH / 2
        else:
            grad = (gk_y - ball_y) / (gk_x - ball_x)
            feature = grad * PITCH_LENGTH / 2 + ball_y - grad * ball_x
        feature = np.asarray([feature])
        return np.clip(feature, -GOAL_WIDTH / 2, GOAL_WIDTH)

    def _ball_feature(self, state):
        ball = np.asarray(self._ball_position(state))
        gk = np.asarray(self._gk_position(state))
        return (ball - gk) / np.linalg.norm(ball - gk)

    def _ball_position(self, state):
        return state[10], state[11]

    def _gk_position(self, state):
        return state[5], state[6]
