# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


class RewardFunction(Model, metaclass=ABCMeta):
    '''Base reward function class
    '''

    def __init__(self, scope_name: str):
        super(RewardFunction, self).__init__(scope_name)

    @abstractmethod
    def r(self, s_current: nn.Variable, a_current: nn.Variable, s_next: nn.Variable) -> nn.Variable:
        '''r
        Computes the reward for the given state, action and next state.
        One (or more than one) of the input variables may not be used in the actual computation.

        Args:
            s_current (nnabla.Variable): State variable
            a_current (nnabla.Variable): Action variable
            s_next (nnabla.Variable): Next state variable

        Returns:
            nnabla.Variable : Reward for the given state, action and next state.
        '''
        raise NotImplementedError
