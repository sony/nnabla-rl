from abc import ABCMeta, abstractmethod

from dataclasses import dataclass, replace

from nnabla_rl.parameter import Parameter
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.exceptions import UnsupportedTrainingException
from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.utils.context as context
import nnabla_rl as rl

import gym


def eval_api(f):
    def wrapped_with_eval_scope(*args, **kwargs):
        with rl.eval_scope():
            return f(*args, **kwargs)
    return wrapped_with_eval_scope


@dataclass
class AlgorithmParam(Parameter):
    pass


class Algorithm(metaclass=ABCMeta):
    def __init__(self, env_info, params=AlgorithmParam()):
        if isinstance(env_info, gym.Env):
            env_info = EnvironmentInfo.from_env(env_info)
        self._env_info = env_info
        self._params = params
        self._iteration_num = 0
        self._max_iterations = 0
        self._hooks = []
        context._set_nnabla_context()

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def latest_iteration_state(self):
        """
        Return latest iteration state that is composed of items of training process state.
        You can use this state for debugging (e.g. plot loss curve).
        See [IterationStateHook](./hooks/iteration_state_hook.py) for getting more details.

        Returns:
            latest_iteration_state (dict): Dictionary with items of training process state.
        """
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        latest_iteration_state['image'] = {}
        return latest_iteration_state

    @property
    def iteration_num(self):
        return self._iteration_num

    @property
    def max_iterations(self):
        return self._max_iterations

    def train(self, env_or_buffer, total_iterations):
        """
        Train the policy with reinforcement learning algorithm

        Args:
            env_or_buffer (gym.Env or ReplayBuffer): Target environment to
                train the policy online or reply buffer to train the policy offline.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException: Raises if this algorithm does not
                support the training method for given parameter.
        """
        if self._is_env(env_or_buffer):
            self.train_online(env_or_buffer, total_iterations)
        elif self._is_buffer(env_or_buffer):
            self.train_offline(env_or_buffer, total_iterations)
        else:
            raise UnsupportedTrainingException

    def train_online(self, train_env, total_iterations):
        """
        Train the policy by interacting with given environment.

        Args:
            train_env (gym.Env): Target environment to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if this algorithm does not support online training
        """
        self._max_iterations = self._iteration_num + total_iterations
        self._before_training_start(train_env)
        while self._iteration_num < self.max_iterations:
            self._iteration_num += 1
            self._run_online_training_iteration(train_env)
            self._invoke_hooks()
        self._after_training_finish(train_env)

    def train_offline(self, replay_buffer, total_iterations):
        """
        Train the policy using only the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer to sample experiences
                to train the policy.
            total_iterations (int): Total number of iterations to train the policy.

        Raises:
            UnsupportedTrainingException:
                Raises if this algorithm does not support offline training
        """
        self._max_iterations = self._iteration_num + total_iterations
        self._before_training_start(replay_buffer)
        while self._iteration_num < self.max_iterations:
            self._iteration_num += 1
            self._run_offline_training_iteration(replay_buffer)
            self._invoke_hooks()
        self._after_training_finish(replay_buffer)

    def set_hooks(self, hooks):
        """
        Set hooks for running additional operation during training.
        Previously set hooks will be removed and replaced with the new hooks.

        Args:
            hooks (list of nnabla_rl.hook.Hook): Hooks to invoke during training
        """
        self._hooks = hooks

    def _invoke_hooks(self):
        for hook in self._hooks:
            hook(self)

    def update_algorithm_params(self, **params):
        self._params = replace(self._params, **params)

    @abstractmethod
    def compute_eval_action(self, state):
        """
        Compute action for given state using current best policy.
        This is used for evaluation.

        Args:
            state (np.ndarray): state to compute the action.

        Returns:
            action (np.ndarray): Best action for given state using current trained policy.
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
        """
        Model objects which are trained by the algorithm.

        Returns:
            models (dict): Dictionary with items of model name as key and object as value.
        """
        raise NotImplementedError

    @abstractmethod
    def _solvers(self):
        """
        Solver objects which are used for training the models by the algorithm.

        Returns:
            solvers (dict): Dictionary with items of solver name as key and object as value.
        """
        raise NotImplementedError

    def _is_env(self, env):
        return isinstance(env, gym.Env)

    def _is_buffer(self, env):
        return isinstance(env, ReplayBuffer)
