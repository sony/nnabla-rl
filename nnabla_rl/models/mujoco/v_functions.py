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

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.v_function import VFunction


class SACVFunction(VFunction):
    '''
    VFunciton model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    '''

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3")
        return h


class TRPOVFunction(VFunction):
    '''
    Vfunction proposed by Peter Henderson, et al.
    in Deep Reinforcement Learning that Matters paper for mujoco environment.
    See: https://arxiv.org/abs/1709.06560.pdf
    '''

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=64, name="linear1",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=64, name="linear2",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
        return h


class PPOVFunction(VFunction):
    '''
    Shared parameter function proposed used in PPO paper for mujoco environment.
    This network outputs the state value
    See: https://arxiv.org/pdf/1707.06347.pdf
    '''

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear1"):
                h = NPF.affine(s, n_outmaps=64,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            with nn.parameter_scope("linear2"):
                h = NPF.affine(h, n_outmaps=64,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1,
                               w_init=RI.NormcInitializer(std=1.0))
        return v


class GAILVFunction(VFunction):
    '''
    Value function proposed by Jonathan Ho, et al.
    See: https://arxiv.org/pdf/1606.03476.pdf
    '''

    def __init__(self, scope_name: str):
        super(GAILVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=100, name="linear1",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=100, name="linear2",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3",
                           w_init=RI.NormcInitializer(std=1.0))
        return h
