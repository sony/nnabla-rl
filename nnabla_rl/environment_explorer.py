# Copyright 2020,2021 Sony Corporation.
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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, cast

import gym
import numpy as np

from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.typing import Action, Experience, State


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
    _state: Union[State, None]
    _action: Union[Action, None]
    _next_state: Union[State, None]
    _steps: int

    def __init__(self,
                 env_info: EnvironmentInfo,
                 config: EnvironmentExplorerConfig = EnvironmentExplorerConfig()):
        self._env_info = env_info
        self._config = config

        self._state = None
        self._action = None
        self._next_state = None
        self._begin_of_episode = True

        self._steps = self._config.initial_step_num

    @abstractmethod
    def action(self, steps: int, state: np.ndarray, *, begin_of_episode: bool = False) -> Tuple[np.ndarray, Dict]:
        '''
        Compute the action for given state at given timestep

        Args:
            steps(int): timesteps since the beginning of exploration
            state(np.ndarray): current state to compute the action
            begin_of_episode(bool): Informs the beginning of episode. Used for rnn state reset.

        Returns:
            np.ndarray: action for current state at given timestep
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
            self._state = cast(State, env.reset())

        for _ in range(n):
            experience, done = self._step_once(env, begin_of_episode=self._begin_of_episode)
            experiences.append(experience)

            self._begin_of_episode = done
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
        self._state = cast(State, env.reset())

        done = False

        experiences = []
        while not done:
            experience, done = self._step_once(env, begin_of_episode=self._begin_of_episode)
            experiences.append(experience)
            self._begin_of_episode = done
        return experiences

    def _step_once(self, env, *, begin_of_episode=False) -> Tuple[Experience, bool]:
        self._steps += 1
        if self._steps < self._config.warmup_random_steps:
            action_info: Dict[str, Any] = {}
            if self._env_info.is_discrete_action_env():
                action = env.action_space.sample()
                self._action = np.asarray(action).reshape((1, ))
            else:
                self._action = env.action_space.sample()
        else:
            self._action, action_info = self.action(self._steps,
                                                    cast(np.ndarray, self._state),
                                                    begin_of_episode=begin_of_episode)

        self._next_state, r, done, step_info = env.step(self._action)
        timelimit = step_info.get('TimeLimit.truncated', False)
        if _is_end_of_episode(done, timelimit, self._config.timelimit_as_terminal):
            non_terminal = 0.0
        else:
            non_terminal = 1.0

        extra_info: Dict[str, Any] = {}
        extra_info.update(action_info)
        extra_info.update(step_info)
        experience = (cast(np.ndarray, self._state),
                      cast(np.ndarray, self._action),
                      r * self._config.reward_scalar,
                      non_terminal,
                      cast(np.ndarray, self._next_state),
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
