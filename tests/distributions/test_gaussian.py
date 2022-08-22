# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

from unittest import mock

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.distributions as D
from nnabla_rl.distributions.gaussian import NnablaGaussian, NumpyGaussian


class TestGaussian():
    def _generate_dummy_mean_var(self):
        batch_size = 10
        output_dim = 10
        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.
        return mean, ln_var

    def test_nnabla_constructor(self):
        mean, ln_var = self._generate_dummy_mean_var()
        distribution = D.Gaussian(nn.Variable.from_numpy_array(mean), nn.Variable.from_numpy_array(ln_var))
        assert isinstance(distribution._delegate, NnablaGaussian)

    def test_nnabla_constructor_with_wrong_shape(self):
        mean, ln_var = self._generate_dummy_mean_var()
        with pytest.raises(AssertionError):
            D.Gaussian(nn.Variable.from_numpy_array(mean[0]), nn.Variable.from_numpy_array(np.diag(ln_var[0])))

    def test_numpy_constructor(self):
        mean, ln_var = self._generate_dummy_mean_var()
        distribution = D.Gaussian(mean[0], np.diag(ln_var[0]))  # without batch
        assert isinstance(distribution._delegate, NumpyGaussian)

    def test_numpy_constructor_with_wrong_shape(self):
        mean, ln_var = self._generate_dummy_mean_var()
        with pytest.raises(AssertionError):
            D.Gaussian(mean, ln_var)

    def test_mix_constructor(self):
        mean, ln_var = self._generate_dummy_mean_var()
        with pytest.raises(ValueError):
            D.Gaussian(nn.Variable.from_numpy_array(mean), ln_var)

        with pytest.raises(ValueError):
            D.Gaussian(mean, nn.Variable.from_numpy_array(ln_var))


class TestNnablaGaussian(object):
    def setup_method(self, method):
        nn.clear_parameters()
        np.random.seed(0)

    def test_sample(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.

        with mock.patch('nnabla_rl.functions.sample_gaussian') as mock_sample_gaussian:
            distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mean),
                                          ln_var=nn.Variable.from_numpy_array(ln_var))
            noise_clip = None
            sampled = distribution.sample(noise_clip=noise_clip)
            sampled.forward()

            assert mock_sample_gaussian.call_count == 1
            args, kwargs = mock_sample_gaussian.call_args

            assert args == (distribution._mean, distribution._ln_var)
            assert kwargs == {"noise_clip": noise_clip}

    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=1.0, stop=2.0, step=0.25))
    def test_sample_and_compute_log_prob(self, mean, var):
        batch_size = 1
        output_dim = 1

        input_shape = (batch_size, output_dim)

        mu = np.ones(shape=input_shape) * mean
        ln_var = np.ones(shape=input_shape) * np.log(var)
        var = np.exp(ln_var)

        distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mu),
                                      ln_var=nn.Variable.from_numpy_array(ln_var))

        sample, log_prob = distribution.sample_and_compute_log_prob()

        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # nnabla's bug?
        nn.forward_all([sample, log_prob])

        x = sample.d
        gaussian_log_prob = -0.5 * np.log(2.0 * np.pi) \
            - 0.5 * ln_var \
            - (x - mu) ** 2 / (2.0 * var)
        expected = np.sum(gaussian_log_prob, axis=-1, keepdims=True)
        actual = log_prob.d

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual, atol=1e-3)

    def test_sample_multiple(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.

        with mock.patch('nnabla_rl.functions.sample_gaussian_multiple') as mock_sample_multiple_gaussian:
            distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mean),
                                          ln_var=nn.Variable.from_numpy_array(ln_var))
            noise_clip = None
            num_samples = 10
            sampled = distribution.sample_multiple(
                num_samples, noise_clip=noise_clip)
            sampled.forward()

            assert mock_sample_multiple_gaussian.call_count == 1

            args, kwargs = mock_sample_multiple_gaussian.call_args
            assert args == (distribution._mean,
                            distribution._ln_var, num_samples)
            assert kwargs == {"noise_clip": noise_clip}

    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=1.0, stop=2.0, step=0.25))
    def test_sample_multiple_and_compute_log_prob(self, mean, var):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)

        mu = np.ones(shape=input_shape) * mean
        ln_var = np.ones(shape=input_shape) * np.log(var)

        distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mu),
                                      ln_var=nn.Variable.from_numpy_array(ln_var))
        num_samples = 10
        samples, log_probs = distribution.sample_multiple_and_compute_log_prob(
            num_samples=num_samples)
        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # nnabla's bug?
        nn.forward_all([samples, log_probs])

        x = samples.d[:, 0, :]
        assert x.shape == (batch_size, output_dim)
        gaussian_log_prob = -0.5 * np.log(2.0 * np.pi) \
            - 0.5 * ln_var \
            - (x - mu) ** 2 / (2.0 * var)
        expected = np.sum(gaussian_log_prob, axis=-1, keepdims=True)
        actual = log_probs.d
        assert expected.shape == (batch_size, 1)
        assert np.allclose(expected, actual[:, 0, :])

        mu = np.reshape(mu, newshape=(batch_size, 1, output_dim))
        ln_var = np.reshape(ln_var, newshape=(
            batch_size, 1, output_dim))

        x = samples.d
        gaussian_log_prob = -0.5 * np.log(2.0 * np.pi) \
            - 0.5 * ln_var \
            - (x - mu) ** 2 / (2.0 * var)
        expected = np.sum(gaussian_log_prob, axis=-1, keepdims=True)

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual, atol=1e-3)

    def test_sample_multiple_and_compute_log_prob_shape(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.

        distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mean),
                                      ln_var=nn.Variable.from_numpy_array(ln_var))
        num_samples = 10
        samples, log_probs = distribution.sample_multiple_and_compute_log_prob(
            num_samples=num_samples)
        nn.forward_all([samples, log_probs])

        assert samples.shape == (batch_size, num_samples, output_dim)
        assert log_probs.shape == (batch_size, num_samples, 1)

    def test_log_prob(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.
        dummy_input = nn.Variable.from_numpy_array(
            np.random.randn(batch_size, output_dim))

        with mock.patch('nnabla_rl.distributions.common_utils.gaussian_log_prob',
                        return_value=nn.Variable.from_numpy_array(np.empty(shape=input_shape))) \
                as mock_gaussian_log_prob:
            distribution = NnablaGaussian(mean=nn.Variable.from_numpy_array(mean),
                                          ln_var=nn.Variable.from_numpy_array(ln_var))
            distribution.log_prob(dummy_input)

            assert mock_gaussian_log_prob.call_count == 1

            args, _ = mock_gaussian_log_prob.call_args
            assert args == (dummy_input,
                            distribution._mean,
                            distribution._var,
                            distribution._ln_var)

    @pytest.mark.parametrize("batch_size", range(1, 10))
    @pytest.mark.parametrize("output_dim", range(1, 10))
    def test_entropy(self, batch_size, output_dim):
        input_shape = (batch_size, output_dim)

        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape)
        ln_var = np.log(sigma) * 2.
        distribution = NnablaGaussian(nn.Variable.from_numpy_array(mean), nn.Variable.from_numpy_array(ln_var))

        actual = distribution.entropy()
        actual.forward()

        assert actual.shape == (batch_size, 1)

        cov = np.diag(sigma[0] ** 2)
        expected = self._gaussian_differential_entropy(covariance_matrix=cov)

        assert np.allclose(expected, actual.d)

    def test_kl_divergence(self):
        batch_size = 10
        output_dim = 10
        input_shape = (batch_size, output_dim)

        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.
        distribution_p = NnablaGaussian(nn.Variable.from_numpy_array(mean), nn.Variable.from_numpy_array(ln_var))
        distribution_q = NnablaGaussian(nn.Variable.from_numpy_array(mean), nn.Variable.from_numpy_array(ln_var))

        actual = distribution_p.kl_divergence(distribution_q)
        actual.forward()

        assert actual.shape == (batch_size, 1)

        expected = np.zeros((batch_size, 1))

        assert expected == pytest.approx(actual.d, abs=1e-5)

    def _gaussian_differential_entropy(self, covariance_matrix):
        # Assuming that covariance_matrix is diagonal
        diagonal = covariance_matrix.diagonal()
        determinant = np.prod(diagonal)

        return 0.5 * np.log(np.power(2.0 * np.pi * np.e, covariance_matrix.shape[0]) * determinant)


class TestNumpyGaussian():
    def _generate_dummy_mean_var(self, scale=5.):
        gaussian_dim = 10
        mean_shape = (gaussian_dim, )
        mean = np.random.normal(size=mean_shape)
        sigma = np.diag(np.ones(shape=mean_shape) * scale)
        sigma_inv = np.diag(1.0 / np.diag(sigma))
        return mean, sigma, sigma_inv

    def test_sample(self):
        mean, sigma, _ = self._generate_dummy_mean_var()
        distribution = NumpyGaussian(mean, np.log(sigma))
        sample = distribution.sample()
        assert sample.shape == mean.shape

        with pytest.raises(NotImplementedError):
            distribution.sample(noise_clip=np.ones(mean.shape))

    def test_numpy_log_prob(self):
        mean, sigma, sigma_inv = self._generate_dummy_mean_var()
        distribution = NumpyGaussian(mean, np.log(sigma))

        query = np.random.normal(size=mean.shape)
        actual = distribution.log_prob(query)

        log_det_term = np.log(np.prod(2.0 * np.pi * np.diag(sigma)))
        quadratic_term = (mean - query).T.dot(sigma_inv).dot(mean - query)
        expected = -0.5 * (log_det_term + quadratic_term)

        assert expected == pytest.approx(actual, abs=1e-5)

    def test_numpy_kl_divergence_different_distribution(self):
        p_mean, p_sigma, _ = self._generate_dummy_mean_var()
        distribution_p = NumpyGaussian(p_mean, np.log(p_sigma))

        q_mean, q_sigma, _ = self._generate_dummy_mean_var(scale=3)
        distribution_q = NumpyGaussian(q_mean, np.log(q_sigma))

        actual = distribution_p.kl_divergence(distribution_q)

        q_sigma_inv = np.diag(1.0 / np.diag(q_sigma))
        trace_term = np.sum(np.diag(q_sigma_inv.dot(p_sigma)))
        quadratic_term = (q_mean - p_mean).T.dot(q_sigma_inv).dot(q_mean - p_mean)
        log_det_term = np.log(np.prod(np.diag(q_sigma)) / np.prod(np.diag(p_sigma)))
        expected = 0.5 * (trace_term + quadratic_term - p_mean.shape[0] + log_det_term)

        assert expected == pytest.approx(actual, abs=1e-5)

    def test_numpy_kl_divergence_identical_distribution(self):
        mean, sigma, _ = self._generate_dummy_mean_var()
        distribution_p = NumpyGaussian(mean, np.log(sigma))
        distribution_q = NumpyGaussian(mean, np.log(sigma))

        actual = distribution_p.kl_divergence(distribution_q)
        expected = np.zeros((1, ))

        assert expected == pytest.approx(actual, abs=1e-5)


if __name__ == '__main__':
    pytest.main()
