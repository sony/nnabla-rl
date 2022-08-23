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

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
from nnabla_rl.models.dynamics import DeterministicDynamics


class MPPIDeterministicDynamics(DeterministicDynamics):
    '''
    MPPI dynamics or classic control discrete environment.
    This network outputs the next state for given state and control input.
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details

    def __init__(self, scope_name: str, dt: float):
        super(MPPIDeterministicDynamics, self).__init__(scope_name=scope_name)
        self._dt = dt

    def next_state(self, x: nn.Variable, u: nn.Variable) -> nn.Variable:
        assert x.shape[-1] % 2 == 0  # must have even number of states (state and its time derivative pairs)
        a = self.acceleration(x, u)
        time_derivatives = NF.concatenate(x[:, x.shape[-1] // 2:], a, axis=len(a.shape)-1)
        return x + time_derivatives * self._dt

    def acceleration(self, x: nn.Variable, u: nn.Variable) -> nn.Variable:
        assert x.shape[-1] % 2 == 0  # must have even number of states (state and its time derivative pairs)
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(x, u, axis=len(u.shape) - 1)
            h = NPF.affine(h, n_outmaps=32, name="linear1")
            h = NF.tanh(h)
            h = NPF.affine(h, n_outmaps=32, name="linear2")
            h = NF.tanh(h)
            a = NPF.affine(h, n_outmaps=x.shape[-1] // 2, name="linear3")
        return a
