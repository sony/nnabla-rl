# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod

import nnabla as nn
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.model import Model


class Dynamics(Model, metaclass=ABCMeta):
    '''Base reward function class
    '''

    def __init__(self, scope_name: str):
        super(Dynamics, self).__init__(scope_name)


class DeterministicDynamics(Dynamics, metaclass=ABCMeta):
    ''' DeterministicDynamics
    Abstract class for deterministic dynamics

    This dynamics returns next state for given state and control input (action).
    '''
    @abstractmethod
    def next_state(self, x: nn.Variable, u: nn.Variable) -> nn.Variable:
        '''next_state

        Args:
            x (nnabla.Variable): State variable
            u (nnabla.Variable): Control input variable

        Returns:
            nnabla.Variable : next state for the given state and control input
        '''
        raise NotImplementedError

    def acceleration(self, x: nn.Variable, u: nn.Variable) -> nn.Variable:
        '''acceleration

        Args:
            x (nnabla.Variable): State variable
            u (nnabla.Variable): Control input variable

        Returns:
            nnabla.Variable: acceleration for the given state and control input
        '''
        raise NotImplementedError


class StochasticDynamics(Dynamics, metaclass=ABCMeta):
    ''' StochasticDynamics
    Abstract class for stochastic dynamics

    This dynamics returns the probability distribution of next state for given state and control input (action).
    '''
    @abstractmethod
    def next_state(self, x: nn.Variable, u: nn.Variable) -> Distribution:
        '''next_state

        Args:
            x (nnabla.Variable): State variable
            u (nnabla.Variable): Control input variable

        Returns:
            nnabla_rl.distributions.Distribution: next state for the given state and control input
        '''
        raise NotImplementedError
