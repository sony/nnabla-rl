# Copyright 2023,2024 Sony Group Corporation.
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
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import DeterministicPolicy


class HyARPolicy(DeterministicPolicy):
    """Actor model proposed by Li et al.

    in the HyAR paper.
    See: https://arxiv.org/abs/2109.05490
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _max_action_value: float

    def __init__(self, scope_name: str, action_dim: int, max_action_value: float):
        super(HyARPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            linear1_init = RI.HeUniform(inmaps=s.shape[1], outmaps=256, factor=1 / 3)
            h = NPF.affine(s, n_outmaps=256, name="linear1", w_init=linear1_init, b_init=linear1_init)
            h = NF.relu(x=h)
            linear2_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
            h = NPF.affine(h, n_outmaps=256, name="linear2", w_init=linear2_init, b_init=linear2_init)
            h = NF.relu(x=h)
            linear3_init = RI.HeUniform(inmaps=h.shape[1], outmaps=self._action_dim, factor=1 / 3)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear3", w_init=linear3_init, b_init=linear3_init)
        return NF.tanh(h) * self._max_action_value
