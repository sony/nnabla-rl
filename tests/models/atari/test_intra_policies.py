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
from nnabla_rl.models.atari.intra_policies import AtariOptionCriticIntraPolicy
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

    def test_intra_pi(self):
        batch_size = 4
        num_dim = 5
        num_options = 4
        action_dim = 3

        s = nn.Variable((batch_size, num_dim))
        option = nn.Variable((batch_size, 1))

        head = DummyModel("shared")
        policy = AtariOptionCriticIntraPolicy(head, scope_name="shared", num_options=num_options, action_dim=action_dim)
        a = policy.intra_pi(s, option)
        p = a._distribution

        input_s = drng.random((4, 5))
        s.d = input_s
        input_option = np.array([[1.0], [0.0], [3.0], [2.0]])
        option.d = input_option
        p.forward()

        params = nn.get_parameters()
        for i in range(batch_size):
            result = (
                np.matmul(input_s[i, np.newaxis], params["shared/linear_intra_pi/w"].d[0, int(input_option[i])])
                + params["shared/linear_intra_pi/b"].d[0, int(input_option[i])]
            )
            expected = self._numpy_softmax(result.flatten())
            assert np.allclose(p.d[i], expected)

    def test_get_parameters(self):
        batch_size = 2
        num_dim = 5
        num_options = 4
        action_dim = 3

        s = nn.Variable((batch_size, num_dim))
        option = nn.Variable((batch_size, 1))

        head = DummyModel("shared")
        policy = AtariOptionCriticIntraPolicy(head, scope_name="shared", num_options=num_options, action_dim=action_dim)
        a = policy.intra_pi(s, option)

        params = policy.get_parameters()

        assert "linear_intra_pi/w" in params.keys()
        assert "linear_intra_pi/b" in params.keys()
        assert "dummy/w" in params.keys()

    def _numpy_softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    pytest.main()
