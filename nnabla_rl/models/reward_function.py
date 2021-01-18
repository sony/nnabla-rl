from abc import ABCMeta, abstractmethod
from typing import Optional, Callable

import nnabla as nn

from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors import Preprocessor


def preprocess_state(function: Callable[[nn.Variable], nn.Variable]) -> Callable[[nn.Variable], nn.Variable]:

    def wrapped(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        if self._state_preprocessor is not None:
            processed = self._state_preprocessor.process(s)
            return function(self, processed, a)
        else:
            raise NotImplementedError("decorated with preprecess_state but preprocessor is not set")
    return wrapped


class RewardFunction(Model, metaclass=ABCMeta):

    _state_preprocessor: Optional[Preprocessor]

    def __init__(self, scope_name: str):
        super(RewardFunction, self).__init__(scope_name)
        self._state_preprocessor = None

    def set_state_preprocessor(self, preprocessor: Preprocessor) -> None:
        self._state_preprocessor = preprocessor

    def has_preprocessor(self) -> bool:
        return self._state_preprocessor is not None

    @abstractmethod
    def r(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        '''r

        Args:
            state (nnabla.Variable): State variable
            action (nnabla.Variable): Action variable


        Returns:
            nnabla.Variable : Reward for the given state and action
        '''
        raise NotImplementedError
