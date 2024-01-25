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

from abc import abstractmethod
from enum import Enum
from typing import Tuple

import gym

from nnabla_rl.external.goal_env import GoalEnv
from nnabla_rl.typing import Experience, Info, NextState, NonTerminal, Reward


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCESS = 1
    FAIL = 2


class AMPEnv(gym.Env):
    def step(self, action):
        next_state, reward, done, info = self._step(action)
        info["task_result"] = self.task_result(next_state, reward, done, info)
        info["valid_episode"] = self.is_valid_episode(next_state, reward, done, info)
        info["expert_experience"] = self.expert_experience(next_state, reward, done, info)
        return next_state, reward, done, info

    @abstractmethod
    def task_result(self, state, reward, done, info) -> TaskResult:
        raise NotImplementedError

    @abstractmethod
    def is_valid_episode(self, state, reward, done, info) -> bool:
        raise NotImplementedError

    @abstractmethod
    def expert_experience(self, state, reward, done, info) -> Experience:
        raise NotImplementedError

    def update_sample_counts(self):
        pass

    @abstractmethod
    def _step(self, action) -> Tuple[NextState, Reward, NonTerminal, Info]:
        raise NotImplementedError("Implement this function for stepping the env and do not override step()")


class AMPGoalEnv(GoalEnv):
    def step(self, action):
        next_state, reward, done, info = self._step(action)
        info["task_result"] = self.task_result(next_state, reward, done, info)
        info["valid_episode"] = self.is_valid_episode(next_state, reward, done, info)
        info["expert_experience"] = self.expert_experience(next_state, reward, done, info)
        return next_state, reward, done, info

    @abstractmethod
    def task_result(self, state, reward, done, info) -> TaskResult:
        raise NotImplementedError

    @abstractmethod
    def is_valid_episode(self, state, reward, done, info) -> bool:
        raise NotImplementedError

    @abstractmethod
    def expert_experience(self, state, reward, done, info) -> Experience:
        raise NotImplementedError

    def update_sample_counts(self):
        pass

    @abstractmethod
    def _step(self, action) -> Tuple[NextState, Reward, NonTerminal, Info]:
        raise NotImplementedError("Implement this function for stepping the env and do not override step()")
