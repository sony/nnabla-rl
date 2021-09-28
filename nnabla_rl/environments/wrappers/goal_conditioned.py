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

import gym


class GoalConditionedTupleObservationEnv(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(GoalConditionedTupleObservationEnv, self).__init__(env)
        self._observation_keys = ['observation', 'desired_goal', 'achieved_goal']

        self._check_env(env)
        self.observation_space = gym.spaces.Tuple([env.observation_space[key]
                                                   for key in self._observation_keys])

    def observation(self, observation):
        self._check_observation(observation)
        return tuple(observation[key].copy() for key in self._observation_keys)

    def _check_env(self, env: gym.Env):
        raw_env = env.unwrapped
        if not issubclass(raw_env.__class__, gym.GoalEnv):
            error_msg = 'This wrapper can take only GoalEnv!!'
            raise ValueError(error_msg)

        for key in env.observation_space.spaces:
            if key not in self._observation_keys:
                error_msg = f'{key} should be included in observation_space!!'
                raise ValueError(error_msg)

    def _check_observation(self, observation):
        for key in observation.keys():
            if key not in self._observation_keys:
                error_msg = f'{key} should be included in observations!!'
                raise ValueError(error_msg)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal=achieved_goal,
                                       desired_goal=desired_goal,
                                       info=info)
