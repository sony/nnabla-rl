# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
import nnabla_rl as rl
import nnabla_rl.functions as RF


class TestFunctions(object):
    def test_sample_gaussian(self):
        batch_size = 1000
        output_dim = 5

        input_shape = (batch_size, output_dim)
        mean = np.ones(shape=input_shape) * 5
        sigma = np.ones(shape=input_shape) * 100
        ln_sigma = np.log(sigma)

        mean_var = nn.Variable(input_shape)
        mean_var.d = mean

        ln_sigma_var = nn.Variable(input_shape)
        ln_sigma_var.d = ln_sigma

        sampled_value = RF.sample_gaussian(
            mean=mean_var, ln_var=(ln_sigma_var * 2.0))
        assert sampled_value.shape == (batch_size, output_dim)

    def test_sample_gaussian_wrong_parameter_shape(self):
        batch_size = 10
        mean_dim = 5
        sigma_dim = 10

        mean = np.ones(shape=(batch_size, mean_dim)) * 5
        sigma = np.ones(shape=(batch_size, sigma_dim)) * 100
        ln_sigma = np.log(sigma)
        mean_var = nn.Variable(mean.shape)
        mean_var.d = mean

        ln_sigma_var = nn.Variable(sigma.shape)
        ln_sigma_var.d = ln_sigma

        with pytest.raises(ValueError):
            RF.sample_gaussian(
                mean=mean_var, ln_var=(ln_sigma_var * 2.0))

    def test_expand_dims(self):
        batch_size = 4
        data_h = 3
        data_w = 2

        data = np.random.normal(size=(batch_size, data_h, data_w))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        actual = RF.expand_dims(data_var, axis=0)
        actual.forward()
        expected = np.expand_dims(data, axis=0)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.expand_dims(data_var, axis=1)
        actual.forward()
        expected = np.expand_dims(data, axis=1)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.expand_dims(data_var, axis=2)
        actual.forward()
        expected = np.expand_dims(data, axis=2)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

    def test_repeat(self):
        batch_size = 4
        data_h = 3
        data_w = 2

        data = np.random.normal(size=(batch_size, data_h, data_w))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        repeats = 5

        actual = RF.repeat(data_var, repeats=repeats, axis=0)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=0)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.repeat(data_var, repeats=repeats, axis=1)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=1)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.repeat(data_var, repeats=repeats, axis=2)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=2)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

    def test_sqrt(self):
        num_samples = 100
        batch_num = 100
        # exp to enforce positive value
        data = np.exp(np.random.normal(size=(num_samples,  batch_num, 1)))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        actual = RF.sqrt(data_var)
        actual.forward(clear_buffer=True)
        expected = np.sqrt(data)
        assert actual.shape == expected.shape
        assert np.all(np.isclose(actual.data.data, expected))

    def test_std(self):
        # stddev computation
        num_samples = 100
        batch_num = 100
        data = np.random.normal(size=(num_samples,  batch_num, 1))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        for axis in range(len(data.shape)):
            actual = RF.std(data_var, axis=axis, keepdims=False)
            actual.forward(clear_buffer=True)
            expected = np.std(data, axis=axis, keepdims=False)
            assert actual.shape == expected.shape
            assert np.all(np.isclose(actual.data.data, expected))

            actual = RF.std(data_var, axis=axis, keepdims=True)
            actual.forward(clear_buffer=True)
            expected = np.std(data, axis=axis, keepdims=True)
            assert actual.shape == expected.shape
            assert np.all(np.isclose(actual.data.data, expected))

    def test_argmax(self):
        num_samples = 100
        batch_num = 100
        data = np.random.normal(size=(num_samples,  batch_num, 1))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        for axis in [None, *range(len(data.shape))]:
            actual = RF.argmax(data_var, axis=axis)
            actual.forward(clear_buffer=True)
            expected = np.argmax(data, axis=axis)
            assert actual.shape == expected.shape
            assert np.all(actual.data.data == expected)

    def test_quantile_huber_loss(self):
        def huber_loss(x0, x1, kappa):
            diff = x0 - x1
            flag = (np.abs(diff) < kappa).astype(np.float32)
            return (flag) * 0.5 * (diff ** 2.0) + (1.0 - flag) * kappa * (np.abs(diff) - 0.5 * kappa)

        def quantile_huber_loss(x0, x1, kappa, tau):
            u = x0 - x1
            delta = np.less(u, np.zeros(shape=u.shape, dtype=np.float32))
            if kappa == 0.0:
                return (tau - delta) * u
            else:
                Lk = huber_loss(x0, x1, kappa=kappa)
                return np.abs(tau - delta) * Lk / kappa

        N = 10
        batch_size = 1
        x0 = np.random.normal(size=(batch_size, N))
        x1 = np.random.normal(size=(batch_size, N))
        for kappa in [0.0, 1.0]:
            tau = np.array([i / N for i in range(1, N + 1)]).reshape((1, -1))
            tau = np.repeat(tau, axis=0, repeats=batch_size)
            tau_var = nn.Variable.from_numpy_array(tau)
            x0_var = nn.Variable.from_numpy_array(x0)
            x1_var = nn.Variable.from_numpy_array(x1)
            loss = RF.quantile_huber_loss(x0=x0_var, x1=x1_var, kappa=kappa, tau=tau_var)
            loss.forward()

            actual = loss.d
            expected = quantile_huber_loss(x0=x0, x1=x1, kappa=kappa, tau=tau)

            assert actual.shape == expected.shape
            assert np.allclose(actual, expected, atol=1e-7)

    def test_mean_squared_error(self):
        num_samples = 100
        batch_num = 100
        # exp to enforce positive value
        x0 = np.exp(np.random.normal(size=(batch_num, num_samples)))
        x0_var = nn.Variable.from_numpy_array(x0)

        x1 = np.exp(np.random.normal(size=(batch_num, num_samples)))
        x1_var = nn.Variable.from_numpy_array(x1)

        actual = RF.mean_squared_error(x0_var, x1_var)
        actual.forward(clear_buffer=True)
        expected = np.mean((x0 - x1)**2)
        assert actual.shape == expected.shape
        assert np.all(np.isclose(actual.d, expected))

    def test_rand_different_seed(self):
        random_value1 = RF.rand(shape=(10, 10))
        random_value1.forward()

        random_value2 = RF.rand(shape=(10, 10))
        random_value2.forward()

        assert not np.allclose(random_value1.d, random_value2.d)

    def test_rand_with_global_seed(self):
        rl.seed(100)
        random_value1 = RF.rand(shape=(10, 10))
        random_value1.forward()

        rl.seed(100)
        random_value2 = RF.rand(shape=(10, 10))
        random_value2.forward()

        assert np.allclose(random_value1.d, random_value2.d)

    def test_randn_different_seed(self):
        random_value1 = RF.randn(shape=(10, 10))
        random_value1.forward()

        random_value2 = RF.randn(shape=(10, 10))
        random_value2.forward()

        assert not np.allclose(random_value1.d, random_value2.d)

    def test_randn_with_global_seed(self):
        rl.seed(100)
        random_value1 = RF.randn(shape=(10, 10))
        random_value1.forward()

        rl.seed(100)
        random_value2 = RF.randn(shape=(10, 10))
        random_value2.forward()

        assert np.allclose(random_value1.d, random_value2.d)

    def test_random_choice_different_seed(self):
        x = nn.Variable.from_numpy_array(np.array([[11, 22, 33], [110, 220, 330]]))
        w = nn.Variable.from_numpy_array(np.array([[10, 20, 70], [70, 20, 10]]))

        random_value1 = RF.random_choice(x, w)
        random_value1.forward()

        random_value2 = RF.random_choice(x, w)
        random_value2.forward()

        assert not np.allclose(random_value1.d, random_value2.d)

    def test_random_choice_with_global_seed(self):
        x = nn.Variable.from_numpy_array(np.array([[11, 22, 33], [110, 220, 330]]))
        w = nn.Variable.from_numpy_array(np.array([[10, 20, 70], [70, 20, 10]]))

        rl.seed(100)
        random_value1 = RF.random_choice(x, w)
        random_value1.forward()

        rl.seed(100)
        random_value2 = RF.random_choice(x, w)
        random_value2.forward()

        assert np.allclose(random_value1.d, random_value2.d)

    def test_gaussian_cross_entropy_method(self):

        def objective_function(x):
            return -((x - 3.)**2)

        batch_size = 1
        var_size = 1

        init_mean = nn.Variable.from_numpy_array(np.zeros((batch_size, var_size)))
        init_var = nn.Variable.from_numpy_array(np.ones((batch_size, var_size)))
        optimal_mean, optimal_top = RF.gaussian_cross_entropy_method(objective_function, init_mean, init_var, alpha=0)

        nn.forward_all([optimal_mean, optimal_top])

        assert np.allclose(optimal_mean.d, np.array([[3.]]), atol=1e-3)
        assert np.allclose(optimal_top.d, np.array([[3.]]), atol=1e-3)

    def test_gaussian_cross_entropy_method_with_complicated_objective_function(self):

        def dummy_q_function(s, a):
            return -((a - s)**2)

        batch_size = 5
        pop_size = 500
        state_size = 1
        action_size = 1

        s = np.arange(batch_size*state_size).reshape(batch_size, state_size)
        s = np.tile(s, (pop_size, 1, 1))
        s = np.transpose(s, (1, 0, 2))
        s_var = nn.Variable.from_numpy_array(s.reshape(batch_size*pop_size, state_size))
        def objective_function(x): return dummy_q_function(s_var, x)

        init_mean = nn.Variable.from_numpy_array(np.zeros((batch_size, action_size)))
        init_var = nn.Variable.from_numpy_array(np.ones((batch_size, action_size))*2)
        optimal_mean, optimal_top = RF.gaussian_cross_entropy_method(
            objective_function, init_mean, init_var, pop_size, alpha=0)

        nn.forward_all([optimal_mean, optimal_top])

        assert np.allclose(optimal_mean.d, np.array([[0.], [1.], [2.], [3.], [4.]]), atol=1e-3)
        assert np.allclose(optimal_top.d, np.array([[0.], [1.], [2.], [3.], [4.]]), atol=1e-3)


if __name__ == "__main__":
    pytest.main()
