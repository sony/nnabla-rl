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
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
from nnabla.parameter import get_parameter_or_create
from nnabla_rl.utils.matrices import compute_hessian


class TestComputeHessian():
    def setup_method(self, method):
        nn.clear_parameters()

    def test_compute_hessian(self):
        x = get_parameter_or_create("x", shape=(1, ))
        y = get_parameter_or_create("y", shape=(1, ))
        loss = x**3 + 2.*x*y + y**2 - x

        x.d = 2.
        y.d = 3.
        actual = compute_hessian(loss, nn.get_parameters().values())

        assert np.array([[12., 2.], [2., 2.]]) == pytest.approx(actual)

    def test_compute_network_parameters(self):
        state = nn.Variable((1, 2))
        output = NPF.affine(state, 1, w_init=NI.ConstantInitializer(
            value=1.), b_init=NI.ConstantInitializer(value=1.))

        loss = NF.sum(output**2)
        state_array = np.array([[1.0, 0.5]])
        state.d = state_array

        actual = compute_hessian(loss, nn.get_parameters().values())

        expected = np.array(
            [[2*state_array[0, 0]**2,
              2*state_array[0, 0]*state_array[0, 1],
              2*state_array[0, 0]],
             [2*state_array[0, 0]*state_array[0, 1],
              2*state_array[0, 1]**2,
              2*state_array[0, 1]],
             [2*state_array[0, 0],
              2*state_array[0, 1],
              2.]]
        )

        assert expected == pytest.approx(actual)
