# Copyright 2024 Sony Group Corporation.
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

from typing import Tuple

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.v_function import VFunction


class AMPVFunction(VFunction):
    """Value function proposed by Xue Bin Peng, et al.

    See: https://arxiv.org/abs/2104.02180
    """

    def __init__(self, scope_name: str):
        super(AMPVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        assert len(s) == 3
        s_for_pi_v, _, _ = s
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(
                s_for_pi_v,
                n_outmaps=1024,
                name="linear1",
                w_init=RI.GlorotUniform(inmaps=s_for_pi_v.shape[1], outmaps=1024),
            )
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=512, name="linear2", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=512))
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=1))
        return h


class AMPGatedVFunction(VFunction):
    """Value function proposed by Xue Bin Peng, et al.

    See: https://arxiv.org/abs/2104.02180
    """

    def __init__(self, scope_name: str):
        super(AMPGatedVFunction, self).__init__(scope_name)

    def v(
        self, s: Tuple[nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable]
    ) -> nn.Variable:
        assert len(s) == 7
        s_for_pi_v, _, _, goal, *_ = s
        with nn.parameter_scope(self.scope_name):
            # gated block
            g_z = NPF.affine(
                goal,
                n_outmaps=128,
                name="gate_initial_linear",
                w_init=RI.GlorotUniform(inmaps=goal.shape[1], outmaps=128),
            )
            g_z = NF.relu(g_z)

            h = NF.concatenate(s_for_pi_v, goal, axis=-1)

            # block1
            h = NPF.affine(h, n_outmaps=1024, name="linear_1", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=1024))

            gate_h = NPF.affine(
                g_z, n_outmaps=64, name="gate_linear1", w_init=RI.GlorotUniform(inmaps=g_z.shape[1], outmaps=64)
            )
            gate_h = NF.relu(gate_h)
            gate_h_bias = NPF.affine(
                gate_h,
                n_outmaps=1024,
                name="gate_bias_linear1",
                w_init=RI.GlorotUniform(inmaps=gate_h.shape[1], outmaps=1024),
            )
            gate_h_scale = NPF.affine(
                gate_h,
                n_outmaps=1024,
                name="gate_scale_linear1",
                w_init=RI.GlorotUniform(inmaps=gate_h.shape[1], outmaps=1024),
            )

            h = h * 2.0 * NF.sigmoid(gate_h_scale) + gate_h_bias
            h = NF.relu(h)

            # block2
            h = NPF.affine(h, n_outmaps=512, name="linear_2", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=512))

            gate_h = NPF.affine(
                g_z, n_outmaps=64, name="gate_linear2", w_init=RI.GlorotUniform(inmaps=g_z.shape[1], outmaps=64)
            )
            gate_h = NF.relu(gate_h)
            gate_h_bias = NPF.affine(
                gate_h,
                n_outmaps=512,
                name="gate_bias_linear2",
                w_init=RI.GlorotUniform(inmaps=gate_h.shape[1], outmaps=512),
            )
            gate_h_scale = NPF.affine(
                gate_h,
                n_outmaps=512,
                name="gate_scale_linear2",
                w_init=RI.GlorotUniform(inmaps=gate_h.shape[1], outmaps=512),
            )

            h = h * 2.0 * NF.sigmoid(gate_h_scale) + gate_h_bias
            h = NF.relu(h)

            # output block
            h = NPF.affine(h, n_outmaps=1, name="linear3", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=1))
        return h
