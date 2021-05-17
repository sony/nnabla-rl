# Copyright 2020,2021 Sony Corporation.
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


class VFunction(Model, metaclass=ABCMeta):
    '''Base Value function class
    '''

    def __init__(self, scope_name: str):
        super(VFunction, self).__init__(scope_name)

    @abstractmethod
    def v(self, s: nn.Variable) -> nn.Variable:
        '''Compute the state value (V) for given state

        Args:
            s (nn.Variable): state variable

        Returns:
            nn.Variable: State value for given state
        '''
        raise NotImplementedError
