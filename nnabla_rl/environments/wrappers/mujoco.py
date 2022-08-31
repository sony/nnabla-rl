# Copyright 2021,2022 Sony Group Corporation.
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

from typing import Dict, Tuple, Union, cast

import gym

from nnabla_rl.environments.gym_utils import extract_max_episode_steps
from nnabla_rl.typing import Action, Info, Reward, State


class EndlessEnv(gym.Wrapper):
    ''' Endless Env
    This environment wrapper makes an environment be endless. \
    Any done flags will be False except for the timelimit,
    and reset_reward (Usually, this value is negative) will be given.
    The done flag is True only if the number of steps reaches the timelimit of the environment.
    '''

    def __init__(self, env: gym.Env, reset_reward: float):
        super(EndlessEnv, self).__init__(env)
        self._fall_done = False
        self._max_episode_steps = extract_max_episode_steps(env)
        self._num_steps = 0
        self._reset_reward = reset_reward

    def step(self, action: Action) -> Union[Tuple[State, Reward, bool, Info],
                                            Tuple[State, Reward, bool, bool, Info]]:
        self._num_steps += 1
        next_state, reward, done, info = cast(Tuple[State, float, bool, Dict], super().step(action))

        timelimit = info.pop('TimeLimit.truncated', False) or (self._num_steps == self._max_episode_steps)

        if timelimit:
            self._num_steps = 0

        if (not timelimit) and done:
            reward = self._reset_reward
            next_state = cast(State, self.reset())

        return next_state, reward, timelimit, info
