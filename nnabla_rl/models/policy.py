from abc import ABCMeta, abstractmethod
from typing import Optional, Callable

import nnabla as nn

from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors import Preprocessor


def preprocess_state(function: Callable[[nn.Variable], nn.Variable]) -> Callable[[nn.Variable], nn.Variable]:
    def wrapped(self, s: nn.Variable) -> nn.Variable:
        if self._state_preprocessor is not None:
            processed = self._state_preprocessor.process(s)
            return function(self, processed)
        else:
            raise NotImplementedError("decorated with preprecess_state but preprocessor is not set")
    return wrapped


class Policy(Model, metaclass=ABCMeta):

    _state_preprocessor: Optional[Preprocessor]

    def __init__(self, scope_name: str):
        super(Policy, self).__init__(scope_name)
        self._state_preprocessor = None

    def set_state_preprocessor(self, preprocessor: Preprocessor) -> None:
        self._state_preprocessor = preprocessor

    def has_preprocessor(self) -> bool:
        return self._state_preprocessor is not None


class DeterministicPolicy(Policy, metaclass=ABCMeta):
    """ DeterministicPolicy
    Abstract class for deterministic policy

    By calling this policy, it will return an action for the given state.
    """
    @abstractmethod
    def pi(self, s: nn.Variable) -> nn.Variable:
        '''pi

        Args:
            state (nnabla.Variable): State variable

        Returns:
            nnabla.Variable : Action for the given state
        '''
        raise NotImplementedError


class StochasticPolicy(Policy, metaclass=ABCMeta):
    ''' StochasticPolicy
    Abstract class for stochastic policy

    By calling this policy, it will return a probability distribution of action for the given state.
    '''
    @abstractmethod
    def pi(self, s: nn.Variable) -> nn.Variable:
        '''pi

        Args:
            state (nnabla.Variable): State variable

        Returns:
            nnabla_rl.distributions.Distribution: Probability distribution of the action for the given state
        '''
        raise NotImplementedError
