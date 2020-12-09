import pytest

import numpy as np

import nnabla as nn

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


if __name__ == "__main__":
    pytest.main()
