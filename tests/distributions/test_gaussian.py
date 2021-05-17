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

from unittest import mock

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.distributions as D


class TestGaussian(object):
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
            distribution = D.Gaussian(mean=mean, ln_var=ln_var)
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

        distribution = D.Gaussian(mean=mu, ln_var=ln_var)

        sample, log_prob = distribution.sample_and_compute_log_prob()

        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # NNabla's bug?
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
            distribution = D.Gaussian(mean=mean, ln_var=ln_var)
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

        distribution = D.Gaussian(mean=mu, ln_var=ln_var)
        num_samples = 10
        samples, log_probs = distribution.sample_multiple_and_compute_log_prob(
            num_samples=num_samples)
        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # NNabla's bug?
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

        distribution = D.Gaussian(mean=mean, ln_var=ln_var)
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
            distribution = D.Gaussian(mean=mean, ln_var=ln_var)
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
        distribution = D.Gaussian(mean, ln_var)

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
        distribution_p = D.Gaussian(mean, ln_var)
        distribution_q = D.Gaussian(mean, ln_var)

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


if __name__ == "__main__":
    pytest.main()
