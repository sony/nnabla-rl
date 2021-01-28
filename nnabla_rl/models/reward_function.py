from abc import ABCMeta, abstractmethod

import nnabla as nn

from nnabla_rl.models.model import Model


class RewardFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(RewardFunction, self).__init__(scope_name)

    @abstractmethod
    def r(self, s_current: nn.Variable, a_current: nn.Variable, s_next: nn.Variable) -> nn.Variable:
        '''r

        Args:
            s_current (nnabla.Variable): State variable
            a_current (nnabla.Variable): Action variable
            s_next (nnabla.Variable): Next state variable


        Returns:
            nnabla.Variable : Reward for the given state and action
        '''
        raise NotImplementedError
