# Copyright 2022 Sony Group Corporation.
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
import nnabla_rl.distributions as D
from nnabla_rl.models import StochasticPolicy


class TestOneHotSoftmax(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_sample(self):
        z = np.array([[0, 0, 1000, 0],
                      [0, 1000, 0, 0],
                      [1000, 0, 0, 0],
                      [0, 0, 0, 1000]])

        batch_size = z.shape[0]
        distribution = D.OneHotSoftmax(z=z)
        sampled = distribution.sample()

        sampled.forward()
        assert sampled.shape == (batch_size, 4)
        assert np.all(sampled.d == np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))

    def test_sample_multi_dimensional(self):
        z = np.array([[[1000, 0, 0, 0],
                       [0, 0, 0, 1000],
                       [0, 1000, 0, 0],
                       [0, 0, 1000, 0]],
                      [[0, 0, 1000, 0],
                       [0, 1000, 0, 0],
                       [1000, 0, 0, 0],
                       [0, 0, 0, 1000]]])
        assert z.shape == (2, 4, 4)
        batch_size = z.shape[0]
        category_size = z.shape[1]
        distribution = D.OneHotSoftmax(z=z)
        sampled = distribution.sample()

        sampled.forward()
        assert sampled.shape == (batch_size, category_size, 4)
        assert np.all(sampled.d == np.array([[[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
                                             [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]]))

    def test_choose_probable(self):
        z = np.array([[[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 1.0, -1.0],
                       [-5.1, 11.2, 0.8, 0.7],
                       [-3.0, -2.1, -7.6, -5.4]],
                      [[0, 0, 1000, 0],
                       [0, 1000, 0, 0],
                       [1000, 0, 0, 0],
                       [0, 0, 0, 1000]]])
        assert z.shape == (2, 4, 4)
        batch_size = z.shape[0]
        category_size = z.shape[1]
        distribution = D.OneHotSoftmax(z=z)
        probable = distribution.choose_probable()

        probable.forward()
        assert probable.shape == (batch_size, category_size, 4)
        assert np.all(probable.d == np.array([[[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                                             [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]]))

    def test_backprop(self):
        class TestModel(StochasticPolicy):
            def pi(self, s: nn.Variable):
                with nn.parameter_scope(self.scope_name):
                    z = NPF.affine(s, n_outmaps=5)
                return D.OneHotSoftmax(z=z)
        model = TestModel('test')

        batch_size = 5
        data_dim = 10
        s = nn.Variable.from_numpy_array(np.ones(shape=(batch_size, data_dim)))
        distribution = model.pi(s)

        for sample_method in (distribution.sample, distribution.choose_probable):
            value = sample_method()
            loss = NF.mean(NF.pow_scalar(value, 2.0))

            for parameter in model.get_parameters().values():
                parameter.grad.zero()

            loss.forward()

            for parameter in model.get_parameters().values():
                assert np.all(parameter.g == 0)

            loss.backward()
            for parameter in model.get_parameters().values():
                assert not np.all(parameter.g == 0)


if __name__ == "__main__":
    pytest.main()
