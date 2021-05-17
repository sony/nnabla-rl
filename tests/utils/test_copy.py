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
import pytest

import nnabla as nn
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
from nnabla_rl.models import Model
from nnabla_rl.utils.misc import copy_network_parameters


class DummyNetwork(Model):
    def __init__(self, scope_name,
                 weight_initializer=None, bias_initialzier=None):
        self._scope_name = scope_name
        self._weight_initializer = weight_initializer
        self._bias_initialzier = bias_initialzier

        super().__init__(scope_name)
        dummy_variable = nn.Variable((1, 1))
        self(dummy_variable)

    def __call__(self, dummy_variable):
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(dummy_variable, 1,
                           w_init=self._weight_initializer,
                           b_init=self._bias_initialzier)
        return h


class TestCopy(object):

    def test_copy_network_parameters(self):
        nn.clear_parameters()

        base = DummyNetwork('base',
                            NI.ConstantInitializer(1),
                            NI.ConstantInitializer(1))
        target = DummyNetwork('target',
                              NI.ConstantInitializer(2),
                              NI.ConstantInitializer(2))

        copy_network_parameters(base.get_parameters(),
                                target.get_parameters(),
                                tau=1.0)

        assert self._has_same_parameters(base.get_parameters(),
                                         target.get_parameters())

    def test_softcopy_network_parameters(self):
        nn.clear_parameters()

        base = DummyNetwork('base',
                            NI.ConstantInitializer(1),
                            NI.ConstantInitializer(1))

        weight_initializer = NI.ConstantInitializer(2)
        bias_initializer = NI.ConstantInitializer(2)
        target_original = DummyNetwork('target_original',
                                       weight_initializer,
                                       bias_initializer)
        target = DummyNetwork('target',
                              weight_initializer,
                              bias_initializer)

        copy_network_parameters(base.get_parameters(),
                                target.get_parameters(),
                                tau=0.75)

        assert self._has_soft_same_parameters(target.get_parameters(),
                                              base.get_parameters(),
                                              target_original.get_parameters(),
                                              tau=0.75)

    def test_softcopy_network_parameters_wrong_tau(self):
        nn.clear_parameters()

        base = DummyNetwork('base')
        target = DummyNetwork('target')

        with pytest.raises(ValueError):
            copy_network_parameters(base.get_parameters(),
                                    target.get_parameters(),
                                    tau=-0.75)

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].d, params2[key].d):
                return False
        return True

    def _has_soft_same_parameters(self, merged_params, base_params1, base_params2, tau):
        for key in merged_params.keys():
            if not np.allclose(merged_params[key].d,
                               base_params1[key].d * tau + base_params2[key].d * (1 - tau)):
                return False
        return True
