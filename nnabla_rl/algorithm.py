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

import sys
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla_rl as rl
from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedEnvironmentException, UnsupportedTrainingException
from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger
from nnabla_rl.model_trainers.model_trainer import ModelTrainer
from nnabla_rl.replay_buffer import ReplayBuffer

F = TypeVar("F", bound=Callable[..., Any])


def eval_api(f: F) -> F:
    def wrapped_with_eval_scope(*args, **kwargs):
        with rl.eval_scope():
            return f(*args, **kwargs)

    return cast(F, wrapped_with_eval_scope)


@dataclass
class AlgorithmConfig(Configuration):
    """List of algorithm common configuration.

    Args:
        gpu_id (int): id of the gpu to use. If negative, the training will run on cpu. Defaults to -1.
    """

    gpu_id: int = -1


class Algorithm(metaclass=ABCMeta):
    """Base Algorithm class.

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
    _hooks: Sequence[Hook]

    def __init__(self, env_info, config=AlgorithmConfig()):
        if isinstance(env_info, gym.Env):
            env_info = EnvironmentInfo.from_env(env_info)
        self._env_info = env_info
        self._config = config
        self._iteration_num = 0
        self._hooks = []

        if not self.is_supported_env(env_info):
            raise UnsupportedEnvironmentException(
                "{} does not support the enviroment. \
                See the algorithm catalog (https://github.com/sony/nnabla-rl/tree/master/nnabla_rl/algorithms) \
                and confirm what kinds of enviroments are supported".format(
                    self.__name__
                )
            )

        if self._config.gpu_id < 0:
            logger.info("algorithm will run on cpu.")
        else:
            logger.info("algorithm will run on gpu: {}".format(self._config.gpu_id))

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def latest_iteration_state(self) -> Dict[str, Any]:
        """Return latest iteration state that is composed of items of training
        process state. You can use this state for debugging (e.g. plot loss
        curve). See [IterationStateHook](./hooks/iteration_state_hook.py) for
        getting more details.

        Returns:
            Dict[str, Any]: Dictionary with items of training process state.
        """
        latest_iteration_state: Dict[str, Any] = {}
        latest_iteration_state["scalar"] = {}
        latest_iteration_state["histogram"] = {}
        latest_iteration_state["image"] = {}
        return latest_iteration_state

    @property
    def iteration_num(self) -> int:
        """Current iteration number.

        Returns:
            int: Current iteration number of running training.
        """
        return self._iteration_num

    def train(self, env_or_buffer: Union[gym.Env, ReplayBuffer], total_iterations: int = sys.maxsize):
        """Train the policy with reinforcement learning algorithm.

        Args:
            env_or_buffer (Union[gym.Env, ReplayBuffer]): Target environment to
                train the policy online or reply buffer to train the policy offline.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException: Raises if this algorithm does not
                support the training method for given parameter.
        """
        if self._is_env(env_or_buffer):
            env_or_buffer = cast(gym.Env, env_or_buffer)
            self.train_online(env_or_buffer, total_iterations)
        elif self._is_buffer(env_or_buffer):
            env_or_buffer = cast(ReplayBuffer, env_or_buffer)
            self.train_offline(env_or_buffer, total_iterations)
        else:
            raise UnsupportedTrainingException

    def train_online(self, train_env: gym.Env, total_iterations: int = sys.maxsize):
        """Train the policy by interacting with given environment.

        Args:
            train_env (gym.Env): Target environment to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if the algorithm does not support online training
        """
        if self._has_rnn_models():
            self._assert_rnn_is_supported()
        self._before_training_start(train_env)
        self._setup_hooks(total_iterations)
        for _ in range(total_iterations):
            self._iteration_num += 1
            self._run_online_training_iteration(train_env)
            self._invoke_hooks()
        self._teardown_hooks(total_iterations)
        self._after_training_finish(train_env)

    def train_offline(self, replay_buffer: ReplayBuffer, total_iterations: int = sys.maxsize):
        """Train the policy using only the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer to sample experiences to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if the algorithm does not support offline training
        """
        if self._has_rnn_models():
            self._assert_rnn_is_supported()
        self._before_training_start(replay_buffer)
        self._setup_hooks(total_iterations)
        for _ in range(total_iterations):
            self._iteration_num += 1
            self._run_offline_training_iteration(replay_buffer)
            self._invoke_hooks()
        self._teardown_hooks(total_iterations)
        self._after_training_finish(replay_buffer)

    def set_hooks(self, hooks: Sequence[Hook]):
        """Set hooks for running additional operation during training.
        Previously set hooks will be removed and replaced with new hooks.

        Args:
            hooks (list of nnabla_rl.hook.Hook): Hooks to invoke during training
        """
        self._hooks = hooks

    def _invoke_hooks(self):
        for hook in self._hooks:
            hook(self)

    def _setup_hooks(self, total_iterations: int):
        for hook in self._hooks:
            hook.setup(self, total_iterations)

    def _teardown_hooks(self, total_iterations: int):
        for hook in self._hooks:
            hook.teardown(self, total_iterations)

    @abstractmethod
    def compute_eval_action(self, state, *, begin_of_episode=False, extra_info={}) -> np.ndarray:
        """Compute action for given state using current best policy. This is
        usually used for evaluation.

        Args:
            state (np.ndarray): state to compute the action.
            begin_of_episode (bool): Used for rnn state resetting. This flag informs the beginning of episode.
            extra_info (Dict[str, Any]): Dictionary to provide extra information to compute the action.
                Most of the algorithm does not use this field.

        Returns:
            np.ndarray: Action for given state using current trained policy.
        """
        raise NotImplementedError

    def compute_trajectory(
        self, initial_trajectory: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
        """Compute trajectory (sequence of state and action tuples) from given
        initial trajectory using current policy. Most of the reinforcement
        learning algorithms does not implement this method. Only the optimal
        control algorithms implements this method.

        Args:
            initial_trajectory (Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]): initial trajectory.

        Returns:
            Tuple[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], Sequence[Dict[str, Any]]]:
                Sequence of state and action tuples and extra information (if exist) at each timestep,
                computed with current best policy. Extra information depends on the algorithm.
                The sequence length is same as the length of initial trajectory.
        """
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
        """Model objects which are trained by the algorithm.

        Returns:
            Dict[str, nnabla_rl.model.Model]: Dictionary with items of model name as key and object as value.
        """
        raise NotImplementedError

    @abstractmethod
    def _solvers(self) -> Dict[str, nn.solver.Solver]:
        """Solver objects which are used for training the models by the
        algorithm.

        Returns:
            Dict[str, nn.solver.Solver]: Dictionary with items of solver name as key and object as value.
        """
        raise NotImplementedError

    def _is_env(self, env):
        return isinstance(env, gym.Env)

    def _is_buffer(self, env):
        return isinstance(env, ReplayBuffer)

    def _has_rnn_models(self):
        for model in self._models().values():
            if model.is_recurrent():
                return True
        return False

    def _assert_rnn_is_supported(self):
        if not self.is_rnn_supported():
            raise RuntimeError(f"{self.__name__} does not support rnn models but rnn models where given!")

    @classmethod
    @abstractmethod
    def is_supported_env(cls, env_or_env_info: Union[gym.Env, EnvironmentInfo]) -> bool:
        """Check whether the algorithm supports the enviroment or not.

        Args:
            env_or_env_info \
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`) \
            : environment or environment info

        Returns:
            bool: True if the algorithm supports the environment. Otherwise False.
        """
        raise NotImplementedError

    @classmethod
    def is_rnn_supported(cls) -> bool:
        """Check whether the algorithm supports rnn models or not.

        Returns:
            bool: True if the algorithm supports rnn models. Otherwise False.
        """
        return False

    @property
    @abstractmethod
    def trainers(self) -> Dict[str, ModelTrainer]:
        raise NotImplementedError
