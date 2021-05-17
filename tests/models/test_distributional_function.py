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
from nnabla_rl.models import ValueDistributionFunction


class ValueDistributionFunctionMock(ValueDistributionFunction):
    def __init__(self, scope_name: str, n_action: int, n_atom: int, v_min: float, v_max: float):
        super(ValueDistributionFunctionMock, self).__init__(scope_name, n_action, n_atom, v_min, v_max)

    def probs(self, *args, **kwargs):
        raise NotImplementedError


class TestValueDistributionFunction(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_compute_z(self):
        scope_name = "test"
        n_action = 4
        n_atom = 100
        v_min = -10
        v_max = 10

        value_distribution_function = ValueDistributionFunctionMock(scope_name, n_action, n_atom, v_min, v_max)
        actual = value_distribution_function._compute_z(n_atom, v_min, v_max)
        actual.forward()

        delta_z = (v_max - v_min) / (n_atom - 1)
        expected = nn.Variable.from_numpy_array(np.asarray([v_min + i * delta_z for i in range(n_atom)]))

        assert expected.shape == actual.shape
        assert np.allclose(expected.d, actual.d)


if __name__ == "__main__":
    pytest.main()
