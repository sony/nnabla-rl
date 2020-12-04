from abc import ABCMeta, abstractmethod

from nnabla_rl.models.model import Model


class QFunction(Model, metaclass=ABCMeta):
    def __call__(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def q(self, s, a):
        raise NotImplementedError

    def max_q(self, s):
        raise NotImplementedError

    def argmax_q(self, s):
        raise NotImplementedError
