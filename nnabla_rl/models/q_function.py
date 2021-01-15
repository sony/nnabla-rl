from abc import ABCMeta, abstractmethod

import nnabla as nn

from nnabla_rl.models.model import Model


class QFunction(Model, metaclass=ABCMeta):
    @abstractmethod
    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def all_q(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def max_q(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def argmax_q(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError
