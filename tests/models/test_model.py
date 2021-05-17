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
import nnabla.parametric_functions as NPF
from nnabla_rl.models.model import Model


class ModelMock(Model):
    def __init__(self, scope_name, input_dim, output_dim):
        super(ModelMock, self).__init__(scope_name)
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __call__(self, s):
        assert s.shape[-1] == self._input_dim
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=10, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=20, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._output_dim, name="linear3")
        return NF.tanh(h)


class TestModel(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_scope_name(self):
        scope_name = "test"
        model = Model(scope_name=scope_name)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        scope_name = "test"
        model = Model(scope_name=scope_name)

        assert len(model.get_parameters()) == 0

    def test_deepcopy_without_model_initialization(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.empty(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        new_scope_name = 'copied'
        copied = model.deepcopy(new_scope_name)
        assert type(copied) is type(model)
        assert len(model.get_parameters()) == 0
        assert len(copied.get_parameters()) == 0

    def test_deepcopy_model_is_same(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.ones(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        # Call once to create params
        model(x)

        new_scope_name = 'copied'
        copied = model.deepcopy(new_scope_name)
        assert type(copied) is type(model)

    def test_deepcopy_model_parameters_are_not_shared(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.ones(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        # Call once to create params
        model(x)

        new_scope_name = 'copied'
        copied = model.deepcopy(new_scope_name)

        for src_value in model.get_parameters().values():
            for dst_value in copied.get_parameters().values():
                assert src_value is not dst_value

    def test_deepcopy_model_has_same_param_num(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.ones(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        # Call once to create params
        expected = model(x)

        new_scope_name = 'copied'
        copied = model.deepcopy(new_scope_name)
        assert len(copied.get_parameters()) == len(model.get_parameters())

        actual = copied(x)

        nn.forward_all([expected, actual])
        # Should output same value
        assert np.allclose(expected.d, actual.d)

    def test_deepcopy_same_scope_name_not_allowed(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.empty(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        # Call once to create params
        model(x)

        with pytest.raises(AssertionError):
            model.deepcopy(scope_name)

    def test_deepcopy_cannot_create_with_existing_scope_name(self):
        scope_name = 'src'
        input_dim = 5
        x = nn.Variable.from_numpy_array(np.empty(shape=(1, input_dim)))
        model = self._create_model_from_input(scope_name=scope_name, x=x)

        # Call once to create params
        model(x)

        new_scope_name = 'new'
        model.deepcopy(new_scope_name)

        # Can not create with same scope twice
        with pytest.raises(RuntimeError):
            model.deepcopy(new_scope_name)

    def _create_model_from_input(self, scope_name, x, output_dim=5):
        input_dim = x.shape[-1]
        return ModelMock(scope_name=scope_name, input_dim=input_dim, output_dim=output_dim)


if __name__ == "__main__":
    pytest.main()
