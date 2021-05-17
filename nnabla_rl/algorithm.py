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
from typing import Any, Dict, Sequence, Union

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers
import nnabla_rl as rl
from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedTrainingException
from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger
from nnabla_rl.replay_buffer import ReplayBuffer


def eval_api(f):
    def wrapped_with_eval_scope(*args, **kwargs):
        with rl.eval_scope():
            return f(*args, **kwargs)
    return wrapped_with_eval_scope


@dataclass
class AlgorithmConfig(Configuration):
    """
    List of algorithm common configuration

    Args:
        gpu_id (int): id of the gpu to use. If negative, the training will run on cpu. Defaults to -1.
    """
    gpu_id: int = -1


class Algorithm(metaclass=ABCMeta):
    """Base Algorithm class

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`)
            : environment or environment info
        config (:py:class:`AlgorithmConfig <nnabla_rl.algorithm.AlgorithmConfig>`):
            configuration of the algorithm

    Note:
        Default functions, solvers and configurations are set to the configurations of each algorithm's original paper.
        Default functions may not work depending on the environment.
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _config: AlgorithmConfig
    _iteration_num: int
    _max_iterations: int
    _hooks: Sequence[Hook]

    def __init__(self, env_info, config=AlgorithmConfig()):
        if isinstance(env_info, gym.Env):
            env_info = EnvironmentInfo.from_env(env_info)
        self._env_info = env_info
        self._config = config
        self._iteration_num = 0
        self._max_iterations = 0
        self._hooks = []

        if self._config.gpu_id < 0:
            logger.info('algorithm will run on cpu')
        else:
            logger.info('algorithm will run on gpu: {}'.format(self._config.gpu_id))

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def latest_iteration_state(self) -> Dict[str, Any]:
        '''
        Return latest iteration state that is composed of items of training process state.
        You can use this state for debugging (e.g. plot loss curve).
        See [IterationStateHook](./hooks/iteration_state_hook.py) for getting more details.

        Returns:
            Dict[str, Any]: Dictionary with items of training process state.
        '''
        latest_iteration_state: Dict[str, Any] = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        latest_iteration_state['image'] = {}
        return latest_iteration_state

    @property
    def iteration_num(self) -> int:
        '''
        Current iteration number.

        Returns:
            int: Current iteration number of running training.
        '''
        return self._iteration_num

    @property
    def max_iterations(self) -> int:
        '''
        Maximum iteration number of running training.

        Returns:
            int: Maximum iteration number of running training.
        '''
        return self._max_iterations

    def train(self, env_or_buffer: Union[gym.Env, ReplayBuffer], total_iterations: int):
        '''
        Train the policy with reinforcement learning algorithm

        Args:
            env_or_buffer (Union[gym.Env, ReplayBuffer]): Target environment to
                train the policy online or reply buffer to train the policy offline.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException: Raises if this algorithm does not
                support the training method for given parameter.
        '''
        if self._is_env(env_or_buffer):
            self.train_online(env_or_buffer, total_iterations)
        elif self._is_buffer(env_or_buffer):
            self.train_offline(env_or_buffer, total_iterations)
        else:
            raise UnsupportedTrainingException

    def train_online(self, train_env: gym.Env, total_iterations: int):
        '''
        Train the policy by interacting with given environment.

        Args:
            train_env (gym.Env): Target environment to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if the algorithm does not support online training
        '''
        self._max_iterations = self._iteration_num + total_iterations
        self._before_training_start(train_env)
        while self._iteration_num < self.max_iterations:
            self._iteration_num += 1
            self._run_online_training_iteration(train_env)
            self._invoke_hooks()
        self._after_training_finish(train_env)

    def train_offline(self, replay_buffer: gym.Env, total_iterations: int):
        '''
        Train the policy using only the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer to sample experiences to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if the algorithm does not support offline training
        '''
        self._max_iterations = self._iteration_num + total_iterations
        self._before_training_start(replay_buffer)
        while self._iteration_num < self.max_iterations:
            self._iteration_num += 1
            self._run_offline_training_iteration(replay_buffer)
            self._invoke_hooks()
        self._after_training_finish(replay_buffer)

    def set_hooks(self, hooks: Sequence[Hook]):
        '''
        Set hooks for running additional operation during training.
        Previously set hooks will be removed and replaced with new hooks.

        Args:
            hooks (list of nnabla_rl.hook.Hook): Hooks to invoke during training
        '''
        self._hooks = hooks

    def _invoke_hooks(self):
        for hook in self._hooks:
            hook(self)

    @abstractmethod
    def compute_eval_action(self, state) -> np.array:
        '''
        Compute action for given state using current best policy.
        This is usually used for evaluation.

        Args:
            state (np.ndarray): state to compute the action.

        Returns:
            np.ndarray: Action for given state using current trained policy.
        '''
        raise NotImplementedError

    def _before_training_start(self, env_or_buffer):
        pass

    @abstractmethod
    def _run_online_training_iteration(self, env):
        raise NotImplementedError

    @abstractmethod
    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _after_training_finish(self, env_or_buffer):
        pass

    @abstractmethod
    def _models(self):
        '''
        Model objects which are trained by the algorithm.

        Returns:
            Dict[str, nnabla_rl.model.Model]: Dictionary with items of model name as key and object as value.
        '''
        raise NotImplementedError

    @abstractmethod
    def _solvers(self) -> Dict[str, nn.solver.Solver]:
        '''
        Solver objects which are used for training the models by the algorithm.

        Returns:
            Dict[str, nn.solver.Solver]: Dictionary with items of solver name as key and object as value.
        '''
        raise NotImplementedError

    def _is_env(self, env):
        return isinstance(env, gym.Env)

    def _is_buffer(self, env):
        return isinstance(env, ReplayBuffer)
