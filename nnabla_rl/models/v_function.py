from abc import ABCMeta, abstractmethod

import nnabla as nn

from nnabla_rl.models.model import Model


class VFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(VFunction, self).__init__(scope_name)

    @abstractmethod
    def v(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError
