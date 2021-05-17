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
from nnabla_rl.utils.multiprocess import (copy_mp_arrays_to_params, copy_params_to_mp_arrays, mp_array_from_np_array,
                                          mp_to_np_array, new_mp_arrays_from_params, np_to_mp_array)


class TestModel(Model):
    def __init__(self, scope_name, input_dim, output_dim,
                 w_init=NI.ConstantInitializer(100.0),
                 b_init=NI.ConstantInitializer(100.0)):
        super(TestModel, self).__init__(scope_name)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, x):
        assert x.shape[1] == self._input_dim

        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(x, n_outmaps=256, name="linear1",
                           w_init=self._w_init, b_init=self._b_init)
            h = NPF.affine(h, n_outmaps=256, name="linear2",
                           w_init=self._w_init, b_init=self._b_init)
            h = NPF.affine(h, n_outmaps=self._output_dim,
                           name="linear3", w_init=self._w_init, b_init=self._b_init)
        return h


class TestMultiprocess(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_mp_array_from_np_array(self):
        np_array = np.empty(shape=(10, 9, 8, 7), dtype=float)
        mp_array = mp_array_from_np_array(np_array)
        assert len(mp_array) == np.prod(np_array.shape)

    def test_mp_to_np_array(self):
        np_array = np.empty(shape=(10, 9, 8, 7), dtype=np.int64)
        mp_array = mp_array_from_np_array(np_array)

        converted = mp_to_np_array(
            mp_array, np_array.shape, dtype=np_array.dtype)

        assert converted.shape == np_array.shape
        assert np.allclose(converted, np_array)

    def test_np_to_mp_array(self):
        np_array = np.random.uniform(size=(10, 9, 8, 7))
        mp_array = mp_array_from_np_array(np_array)

        test_array = np.random.uniform(size=(10, 9, 8, 7))
        before_copying = mp_to_np_array(
            mp_array, test_array.shape, dtype=test_array.dtype)
        assert not np.allclose(before_copying, test_array)

        mp_array = np_to_mp_array(test_array, mp_array, dtype=test_array.dtype)
        after_copying = mp_to_np_array(
            mp_array, test_array.shape, dtype=test_array.dtype)
        assert np.allclose(after_copying, test_array)

    def test_new_mp_arrays_from_params(self):
        model = TestModel(scope_name="test", input_dim=5, output_dim=5)
        state = nn.Variable.from_numpy_array(np.empty(shape=(1, 5)))
        model(state)
        params = model.get_parameters()
        mp_arrays = new_mp_arrays_from_params(params)

        for key, value in params.items():
            assert key in mp_arrays
            mp_array = mp_arrays[key]
            print('key: ', key)
            assert len(mp_array) == len(value.d.flatten())

    def test_copy_params_to_mp_arrays(self):
        model = TestModel(scope_name="test", input_dim=5, output_dim=5)
        state = nn.Variable.from_numpy_array(np.empty(shape=(1, 5)))
        model(state)
        params = model.get_parameters()
        mp_arrays = new_mp_arrays_from_params(params)

        for value in mp_arrays.values():
            value[:] = 1.0
            assert not np.allclose(value, 100.0)

        copy_params_to_mp_arrays(params, mp_arrays)

        for value in mp_arrays.values():
            assert np.allclose(value, 100.0)

    def test_copy_mp_arrays_to_params(self):
        model = TestModel(scope_name="test", input_dim=5, output_dim=5)
        state = nn.Variable.from_numpy_array(np.empty(shape=(1, 5)))
        model(state)
        params = model.get_parameters()
        mp_arrays = new_mp_arrays_from_params(params)

        for mp_array in mp_arrays.values():
            mp_array[:] = 50.0

        for value in params.values():
            assert not np.allclose(value.d, 50.0)

        copy_mp_arrays_to_params(mp_arrays, params)

        for value in params.values():
            assert np.allclose(value.d, 50.0)


if __name__ == '__main__':
    pytest.main()
