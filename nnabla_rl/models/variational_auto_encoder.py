from abc import ABCMeta, abstractmethod

from nnabla_rl.models.model import Model


class VariationalAutoEncoder(Model, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError

    @abstractmethod
    def encode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args):
        raise NotImplementedError

    @abstractmethod
    def decode_multiple(self, decode_num, *args):
        raise NotImplementedError

    @abstractmethod
    def latent_distribution(self, *args):
        raise NotImplementedError
