from abc import ABCMeta, abstractmethod

from nnabla_rl.models.model import Model


class Perturbator(Model, metaclass=ABCMeta):
    ''' DeterministicPolicy
    Abstract class for perturbator

    Perturbator generates noise to append to current state's action
    '''

    def __init__(self, scope_name):
        super(Perturbator, self).__init__(scope_name)

    @abstractmethod
    def generate_noise(self, s, a, phi):
        raise NotImplementedError
