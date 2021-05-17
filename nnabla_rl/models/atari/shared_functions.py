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
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.model import Model


class PPOSharedFunctionHead(Model):
    def __init__(self, scope_name, state_shape, action_dim):
        super(PPOSharedFunctionHead, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._action_dim = action_dim

    def __call__(self, s):
        assert s.shape[1:] == self._state_shape
        batch_size = s.shape[0]

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, kernel=(8, 8), stride=(4, 4),
                                    w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2),
                                    w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1),
                                    w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
            with nn.parameter_scope("linear1"):
                h = NPF.affine(h, n_outmaps=512,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
        return h


class A3CSharedFunctionHead(Model):
    def __init__(self, scope_name, state_shape):
        super(A3CSharedFunctionHead, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape

    def __call__(self, s):
        assert s.shape[1:] == self._state_shape
        batch_size = s.shape[0]

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=16, kernel=(8, 8), stride=(4, 4),
                                    w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=32, kernel=(4, 4), stride=(2, 2),
                                    w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
            with nn.parameter_scope("linear1"):
                h = NPF.affine(h, n_outmaps=256,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.relu(x=h)
        return h
