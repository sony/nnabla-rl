from abc import ABCMeta, abstractmethod

import nnabla as nn

from nnabla_rl.models.model import Model


class Policy(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(Policy, self).__init__(scope_name)


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
