# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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
from nnabla_rl.preprocessors.running_mean_normalizer import RunningMeanNormalizer


class TestRunningMeanNormalizer():
    def setup_method(self, method):
        nn.clear_parameters()
        np.random.seed(0)

    @pytest.mark.parametrize("x1, x2, x3",
                             [(np.random.randn(1, 3), np.random.randn(1, 3), np.random.randn(1, 3)),
                              (np.random.randn(1, 2), np.random.randn(2, 2), np.random.randn(3, 2))])
    def test_update(self, x1, x2, x3):
        state_dim = x1.shape[1]
        normalizer = RunningMeanNormalizer(
            scope_name="test", shape=(state_dim, ), epsilon=0.0)

        normalizer.update(x1)
        normalizer.update(x2)
        normalizer.update(x3)

        concat_array = np.concatenate([x1, x2, x3], axis=0)
        expected_mean = np.mean(concat_array, axis=0)
        expected_var = np.var(concat_array, axis=0)

        assert np.allclose(expected_mean, normalizer._mean.d, atol=1e-4)
        assert np.allclose(expected_var, normalizer._var.d, atol=1e-4)

    @pytest.mark.parametrize("mean, var, s_batch",
                             [(np.ones((1, 3)), np.ones((1, 3))*0.2, np.random.randn(1, 3)),
                              (np.ones((1, 2))*0.5, np.ones((1, 2))*0.1, np.random.randn(3, 2))])
    def test_filter(self, mean, var, s_batch):
        state_dim = s_batch.shape[1]
        normalizer = RunningMeanNormalizer(
            scope_name="test", shape=(state_dim, ), epsilon=0.0)

        normalizer._mean.d = mean
        normalizer._var.d = var

        # build computational graph
        s_batch_var = nn.Variable(shape=s_batch.shape)
        filtered_s_batch = normalizer.process(s_batch_var)

        s_batch_var.d = s_batch
        filtered_s_batch.forward()

        actual = (s_batch - mean) / np.sqrt(var + 1e-8)

        assert np.allclose(filtered_s_batch.d, actual, atol=1e-4)

    def test_invalid_value_clip(self):
        with pytest.raises(ValueError):
            RunningMeanNormalizer("test", (1, 1), value_clip=[5., -5.])

    def test_numpy_initializer(self):
        shape = (6, )
        mean_initializer = np.random.rand(6)
        var_initializer = np.random.rand(6)
        normalizer = RunningMeanNormalizer(scope_name="test", shape=shape, epsilon=0.0,
                                           mean_initializer=mean_initializer, var_initializer=var_initializer)

        # dummy process
        output = normalizer.process(nn.Variable.from_numpy_array(np.random.rand(1, 6)))
        output.forward()

        actual_params = normalizer.get_parameters()
        assert np.allclose(actual_params["mean"].d, mean_initializer[np.newaxis, :])
        assert np.allclose(actual_params["var"].d, var_initializer[np.newaxis, :])
        # count should be default initial value
        assert np.allclose(actual_params["count"].d, np.ones((1, 1)) * 1e-4)

    def test_nnabla_initializer(self):
        shape = (6, )
        mean_initializer = NI.ConstantInitializer(5.0)
        var_initializer = NI.ConstantInitializer(6.0)
        normalizer = RunningMeanNormalizer(scope_name="test", shape=shape, epsilon=0.0,
                                           mean_initializer=mean_initializer, var_initializer=var_initializer)

        # dummy process
        output = normalizer.process(nn.Variable.from_numpy_array(np.random.rand(1, 6)))
        output.forward()

        actual_params = normalizer.get_parameters()
        assert np.allclose(actual_params["mean"].d, np.ones((1, 6)) * 5.0)
        assert np.allclose(actual_params["var"].d, np.ones((1, 6)) * 6.0)
        # count should be default initial value
        assert np.allclose(actual_params["count"].d, np.ones((1, 1)) * 1e-4)

    def test_numpy_initializer_with_invalid_mean_initializer_shape(self):
        shape = (6, )
        mean_initializer = np.random.rand(4)
        var_initializer = np.random.rand(6)
        with pytest.raises(AssertionError):
            RunningMeanNormalizer(scope_name="test", shape=shape, epsilon=0.0,
                                  mean_initializer=mean_initializer, var_initializer=var_initializer)

    def test_numpy_initializer_with_invalid_var_initializer_shape(self):
        shape = (6, )
        mean_initializer = np.random.rand(6)
        var_initializer = np.random.rand(4)
        with pytest.raises(AssertionError):
            RunningMeanNormalizer(scope_name="test", shape=shape, epsilon=0.0,
                                  mean_initializer=mean_initializer, var_initializer=var_initializer)
