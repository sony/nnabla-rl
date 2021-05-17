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

import nnabla as nn
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.atari.shared_functions import PPOSharedFunctionHead
from nnabla_rl.models.v_function import VFunction


class PPOVFunction(VFunction):
    '''
    Shared parameter function proposed used in PPO paper for atari environment.
    This network outputs the value
    See: https://arxiv.org/pdf/1707.06347.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _head: PPOSharedFunctionHead

    def __init__(self, head: PPOSharedFunctionHead, scope_name: str):
        super(PPOVFunction, self).__init__(scope_name=scope_name)
        self._head = head

    def v(self, s: nn.Variable) -> nn.Variable:
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1,
                               w_init=RI.NormcInitializer(std=0.01))
        return v

    def _hidden(self, s: nn.Variable) -> nn.Variable:
        return self._head(s)


class A3CVFunction(VFunction):
    '''
    Shared parameter function proposed and used in A3C paper for atari environment.
    See: https://arxiv.org/pdf/1602.01783.pdf
    '''

    def __init__(self, head, scope_name, state_shape):
        super(A3CVFunction, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._head = head

    def v(self, s):
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1)
        return v

    def _hidden(self, s):
        assert s.shape[1:] == self._state_shape
        return self._head(s)
