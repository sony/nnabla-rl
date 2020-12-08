from typing import List

from abc import abstractmethod, ABCMeta

from gym import Env

import numpy as np

from dataclasses import dataclass

from nnabla_rl.parameter import Parameter
from nnabla_rl.environments.environment_info import EnvironmentInfo


@dataclass
class EnvironmentExplorerParam(Parameter):
    warmup_random_steps: int = 0
    reward_scalar: float = 1.0
    timelimit_as_terminal: bool = True
    initial_step_num: int = 0


class EnvironmentExplorer(metaclass=ABCMeta):
    def __init__(self,
                 env_info: EnvironmentInfo,
                 params: EnvironmentExplorerParam = EnvironmentExplorerParam()):
        self._env_info = env_info
        self._params = params

        self._state = None
        self._action = None
        self._next_state = None

        self._steps = self._params.initial_step_num

    @abstractmethod
    def action(self, steps, state):
        raise NotImplementedError

    def step(self, env: Env, n: int = 1) -> List:
        assert 0 < n
        experiences = []
        if self._state is None:
            self._state = env.reset()

        for _ in range(n):
            experience, _ = self._step_once(env)
            experiences.append(experience)
        return experiences

    def rollout(self, env) -> List:
        self._state = env.reset()

        done = False

        experiences = []
        while not done:
            experience, done = self._step_once(env)
            experiences.append(experience)
        return experiences

    def _step_once(self, env):
        self._steps += 1
        if self._steps < self._params.warmup_random_steps:
            action_info = {}
            if self._env_info.is_discrete_action_env():
                action = env.action_space.sample()
                self._action = np.asarray(action).reshape((1, ))
            else:
                self._action = env.action_space.sample()
        else:
            self._action, action_info = self.action(self._steps, self._state)

        self._next_state, r, done, step_info = env.step(self._action)
        truncated = step_info.get('TimeLimit.truncated', False) and self._params.timelimit_as_terminal
        if done and not truncated:
            non_terminal = 0.0
        else:
            non_terminal = 1.0

        additional_info = {}
        additional_info.update(action_info)
        additional_info.update(step_info)
        experience = (self._state, self._action, r * self._params.reward_scalar,
                      non_terminal, self._next_state, additional_info)

        if done:
            self._state = env.reset()
        else:
            self._state = self._next_state

        return experience, done
