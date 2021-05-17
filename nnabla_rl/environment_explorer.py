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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import gym
import numpy as np

from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.typing import Experience


@dataclass
class EnvironmentExplorerConfig(Configuration):
    warmup_random_steps: int = 0
    reward_scalar: float = 1.0
    timelimit_as_terminal: bool = True
    initial_step_num: int = 0


class EnvironmentExplorer(metaclass=ABCMeta):
    '''
    Base class for environment exploration methods.
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _config: EnvironmentExplorerConfig
    _state: Optional[np.array]
    _action: Optional[np.array]
    _next_state: Optional[np.array]
    _steps: int

    def __init__(self,
                 env_info: EnvironmentInfo,
                 config: EnvironmentExplorerConfig = EnvironmentExplorerConfig()):
        self._env_info = env_info
        self._config = config

        self._state = None
        self._action = None
        self._next_state = None

        self._steps = self._config.initial_step_num

    @abstractmethod
    def action(self, steps: int, state: np.array) -> np.array:
        '''
        Compute the action for given state at given timestep

        Args:
            steps(int): timesteps since the beginning of exploration
            state(np.array): current state to compute the action

        Returns:
            np.array: action for current state at given timestep
        '''
        raise NotImplementedError

    def step(self, env: gym.Env, n: int = 1, break_if_done: bool = False) -> List[Experience]:
        '''
        Step n timesteps in given env

        Args:
            env(gym.Env): Environment
            n(int): Number of timesteps to act in the environment

        Returns:
            List[Experience]: List of experience.
                Experience consists of (state, action, reward, terminal flag, next state and extra info).
        '''
        assert 0 < n
        experiences = []
        if self._state is None:
            self._state = env.reset()

        for _ in range(n):
            experience, done = self._step_once(env)
            experiences.append(experience)

            if done and break_if_done:
                break
        return experiences

    def rollout(self, env: gym.Env) -> List[Experience]:
        '''
        Rollout the episode in current env

        Args:
            env(gym.Env): Environment

        Returns:
            List[Experience]: List of experience.
                Experience consists of (state, action, reward, terminal flag, next state and extra info).
        '''
        self._state = env.reset()

        done = False

        experiences = []
        while not done:
            experience, done = self._step_once(env)
            experiences.append(experience)
        return experiences

    def _step_once(self, env) -> Tuple[Experience, bool]:
        self._steps += 1
        if self._steps < self._config.warmup_random_steps:
            action_info = {}
            if self._env_info.is_discrete_action_env():
                action = env.action_space.sample()
                self._action = np.asarray(action).reshape((1, ))
            else:
                self._action = env.action_space.sample()
        else:
            self._action, action_info = self.action(self._steps, self._state)

        self._next_state, r, done, step_info = env.step(self._action)
        timelimit = step_info.get('TimeLimit.truncated', False)
        if _is_end_of_episode(done, timelimit, self._config.timelimit_as_terminal):
            non_terminal = 0.0
        else:
            non_terminal = 1.0

        extra_info: Dict[str, Any] = {}
        extra_info.update(action_info)
        extra_info.update(step_info)
        experience = (cast(np.array, self._state),
                      cast(np.array, self._action),
                      r * self._config.reward_scalar,
                      non_terminal,
                      cast(np.array, self._next_state),
                      extra_info)

        if done:
            self._state = env.reset()
        else:
            self._state = self._next_state

        return experience, done


def _is_end_of_episode(done, timelimit, timelimit_as_terminal):
    if not done:
        return False
    else:
        return (not timelimit) or (timelimit and timelimit_as_terminal)
