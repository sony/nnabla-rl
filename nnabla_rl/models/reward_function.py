from abc import ABCMeta, abstractmethod

import nnabla as nn

from nnabla_rl.models.model import Model


class RewardFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(RewardFunction, self).__init__(scope_name)

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
