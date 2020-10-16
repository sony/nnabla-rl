from abc import ABCMeta, abstractmethod

from nnabla_rl.models.model import Model


class QFunction(Model, metaclass=ABCMeta):
    def __call__(self, s, a):
        raise NotImplementedError

    @abstractmethod
    def q(self, s, a):
        raise NotImplementedError

    def maximum(self, s):
        raise NotImplementedError

    def argmax(self, s):
        raise NotImplementedError
