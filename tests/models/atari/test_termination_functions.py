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

import numpy as np
import pytest

import nnabla as nn
import nnabla.initializer as NI
from nnabla_rl.models.atari.termination_functions import AtariOptionCriticTerminationFunction
from nnabla_rl.models.model import Model
from nnabla_rl.random import drng


class DummyModel(Model):
    def __init__(self, scope_name: str):
        super().__init__(scope_name)

    def __call__(self, s: nn.Variable):
        with nn.parameter_scope(self._scope_name):
            w = nn.parameter.get_parameter_or_create("dummy/w", shape=(1, 1), initializer=NI.ConstantInitializer(1.0))
            s = w * s
        return s


class TestAtariOptionCriticIntraPolicy:
    def setup_method(self, method):
        nn.clear_parameters()

    def test_termination(self):
        batch_size = 2
        num_dim = 5
        num_options = 4

        s = nn.Variable((batch_size, num_dim))
        option = nn.Variable((batch_size, 1))

        head = DummyModel("shared")
        termination = AtariOptionCriticTerminationFunction(head, scope_name="termination", num_options=num_options)
        t = termination.termination(s, option)
        p = t.mean()

        input_s = drng.random((2, 5))
        s.d = input_s
        input_option = np.array([[1.0], [3.0]])
        option.d = input_option
        p.forward()

        params = nn.get_parameters()
        result = (
            np.matmul(input_s, params["termination/linear_termination/affine/W"].d)
            + params["termination/linear_termination/affine/b"].d
        )
        expected_batch = self._numpy_sigmoid(
            np.array([result[0, int(input_option[0])], result[1, int(input_option[1])]])
        )
        assert np.allclose(p.d.flatten(), expected_batch.flatten())

    def test_get_parameters(self):
        batch_size = 2
        num_dim = 5
        num_options = 4

        s = nn.Variable((batch_size, num_dim))
        option = nn.Variable((batch_size, 1))

        head = DummyModel("shared")
        termination = AtariOptionCriticTerminationFunction(head, scope_name="termination", num_options=num_options)
        t = termination.termination(s, option)

        params = termination.get_parameters()

        assert "linear_termination/affine/W" in params.keys()
        assert "linear_termination/affine/b" in params.keys()
        # if the scope name of the head is different, then the head parameters should not be included.
        assert (not "dummy/w" in params.keys()) and ("shared/dummy/w" in nn.get_parameters())

    def _numpy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    pytest.main()
