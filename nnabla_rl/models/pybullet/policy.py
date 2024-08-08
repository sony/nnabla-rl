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

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.policy import StochasticPolicy


class AMPPolicy(StochasticPolicy):
    """Actor model proposed by Xue Bin Peng, et al.

    in AMP paper for their bullet environment.
    This network outputs the policy distribution
    See: https://arxiv.org/abs/2104.02180
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _output_layer_initializer_scale: float

    def __init__(self, scope_name: str, action_dim: int, output_layer_initializer_scale: float):
        super(AMPPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        assert output_layer_initializer_scale > 0.0, f"{output_layer_initializer_scale} should be larger than 0.0"
        self._output_layer_initializer_scale = output_layer_initializer_scale

    def pi(self, s: Tuple[nn.Variable, nn.Variable, nn.Variable]) -> Distribution:
        assert len(s) == 3
        s_for_pi_v, _, _ = s
        batch_size = s_for_pi_v.shape[0]
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
            mean = NPF.affine(
                h,
                n_outmaps=self._action_dim,
                name="linear3_mean",
                w_init=NI.UniformInitializer(
                    (-1.0 * self._output_layer_initializer_scale, self._output_layer_initializer_scale)
                ),
                b_init=NI.ConstantInitializer(0.0),
            )
            ln_sigma = nn.Variable.from_numpy_array(
                np.ones((batch_size, self._action_dim), dtype=np.float32) * np.log(0.05)
            )
            ln_var = ln_sigma * 2.0
            assert mean.shape == ln_var.shape
            assert mean.shape == (s_for_pi_v.shape[0], self._action_dim)
        return D.Gaussian(mean=mean, ln_var=ln_var)


class AMPGatedPolicy(StochasticPolicy):
    """Actor model proposed by Xue Bin Peng, et al.

    in AMP paper for their bullet environment.
    This network outputs the policy distribution
    See: https://arxiv.org/abs/2104.02180
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _output_layer_initializer_scale: float

    def __init__(self, scope_name: str, action_dim: int, output_layer_initializer_scale: float):
        super(AMPGatedPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        assert output_layer_initializer_scale > 0.0, f"{output_layer_initializer_scale} should be larger than 0.0"
        self._output_layer_initializer_scale = output_layer_initializer_scale

    def pi(
        self, s: Tuple[nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable, nn.Variable]
    ) -> Distribution:
        assert len(s) == 7
        s_for_pi_v, _, _, goal, *_ = s
        batch_size = s_for_pi_v.shape[0]
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
                g_z, n_outmaps=64, name="gate_linear2", w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
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
            mean = NPF.affine(
                h,
                n_outmaps=self._action_dim,
                name="linear3_mean",
                w_init=NI.UniformInitializer(
                    (-1.0 * self._output_layer_initializer_scale, self._output_layer_initializer_scale)
                ),
                b_init=NI.ConstantInitializer(0.0),
            )
            ln_sigma = nn.Variable.from_numpy_array(
                np.ones((batch_size, self._action_dim), dtype=np.float32) * np.log(0.05)
            )
            ln_var = ln_sigma * 2.0
            assert mean.shape == ln_var.shape
            assert mean.shape == (batch_size, self._action_dim)

        return D.Gaussian(mean=mean, ln_var=ln_var)
