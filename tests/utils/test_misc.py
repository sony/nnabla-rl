# Copyright 2021,2022,2023 Sony Group Corporation.
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
from nnabla_rl.utils.misc import create_attention_mask, create_variable


class TestMisc():
    def test_create_variable_int(self):
        batch_size = 3
        shape = 5
        actual_var = create_variable(batch_size, shape)

        assert actual_var.shape == (batch_size, shape)

    def test_create_variable_tuple(self):
        batch_size = 3
        shape = (5, 6)
        actual_var = create_variable(batch_size, shape)

        assert actual_var.shape == (batch_size, *shape)

    def test_create_variable_tuples(self):
        batch_size = 3
        shape = ((6, ), (3, ))
        actual_var = create_variable(batch_size, shape)

        assert isinstance(actual_var, tuple)
        assert actual_var[0].shape == (batch_size, *shape[0])
        assert actual_var[1].shape == (batch_size, *shape[1])

    def test_create_attention_mask(self):
        num_query = 3
        num_key = 3

        expected_mask = np.ones(shape=(1, num_query, num_key))
        expected_mask = (1.0 - np.tril(expected_mask)) * np.finfo(np.float32).min
        expected_mask = np.reshape(expected_mask, newshape=(1, *expected_mask.shape))
        expected_mask = nn.Variable.from_numpy_array(expected_mask)
        actual_mask = create_attention_mask(num_query, num_key)
        assert np.allclose(expected_mask.d, actual_mask.d)


if __name__ == "__main__":
    pytest.main()
