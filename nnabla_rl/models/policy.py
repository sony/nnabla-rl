# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from nnabla_rl.models.model import Model


class Policy(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(Policy, self).__init__(scope_name)


class DeterministicPolicy(Policy, metaclass=ABCMeta):
    ''' DeterministicPolicy
    Abstract class for deterministic policy

    By calling this policy, it will return an action for the given state.
    '''
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
