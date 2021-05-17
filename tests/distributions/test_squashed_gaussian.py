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
import nnabla_rl.distributions as D


class TestSquashedGaussian(object):
    def setup_method(self, method):
        nn.clear_parameters()
        np.random.seed(0)

    def test_sample(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5
        ln_var = np.log(sigma) * 2.0

        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        sampled = distribution.sample()
        assert sampled.shape == input_shape

        sampled.forward(clear_no_need_grad=True)
        sampled = sampled.data.data
        assert np.alltrue((-1.0 <= sampled) & (sampled <= 1.0))

    def test_choose_probable(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape)
        ln_var = np.log(sigma) * 2.0

        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        probable_action = distribution.choose_probable()
        assert probable_action.shape == mean.shape

        probable_action.forward(clear_no_need_grad=True)
        probable_action = probable_action.data.data
        assert np.alltrue((-1.0 <= probable_action) & (probable_action <= 1.0))
        assert np.allclose(probable_action, np.tanh(mean), atol=1e-5)

    def test_mean(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape)
        ln_var = np.log(sigma) * 2.0

        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        with pytest.raises(NotImplementedError):
            distribution.mean()

    @pytest.mark.parametrize("x", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=1.0, stop=2.0, step=0.25))
    def test_log_prob(self, x, mean, var):
        mean = np.array(mean).reshape((1, 1))
        ln_var = np.array(np.log(var)).reshape((1, 1))
        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        ln_var = np.log(var)
        gaussian_log_prob = -0.5 * \
            np.log(2.0 * np.pi) - 0.5 * ln_var - \
            (x - mean) ** 2 / (2.0 * var)
        log_det_jacobian = np.log(1 - np.tanh(x) ** 2)
        expected = np.sum(gaussian_log_prob -
                          log_det_jacobian, axis=-1, keepdims=True)

        x_var = nn.Variable((1, 1))
        x_var.d = np.tanh(x)
        actual = distribution.log_prob(x_var)
        actual.forward(clear_no_need_grad=True)
        actual = actual.data.data
        assert np.isclose(expected, actual)

    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=0.5, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=0.5, stop=1.5, step=0.25))
    def test_sample_and_compute_log_prob(self, mean, var):
        mean = np.array(mean).reshape((1, 1))
        ln_var = np.array(np.log(var)).reshape((1, 1))
        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        ln_var = np.log(var)

        sample, actual = distribution.sample_and_compute_log_prob()
        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # NNabla's bug?
        nn.forward_all([sample, actual])

        x = np.arctanh(sample.data.data, dtype=np.float64)
        gaussian_log_prob = -0.5 * \
            np.log(2.0 * np.pi) - 0.5 * ln_var - \
            (x - mean) ** 2 / (2.0 * var)
        log_det_jacobian = np.log(1 - np.tanh(x) ** 2)
        expected = np.sum(gaussian_log_prob - log_det_jacobian, axis=-1, keepdims=True)

        actual = actual.data.data
        assert np.isclose(expected, actual, atol=1e-3, rtol=1e-2)

    def test_sample_and_compute_log_prob_shape(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape)
        ln_var = np.log(sigma) * 2.0

        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)

        sample, actual_log_prob = distribution.sample_and_compute_log_prob()
        assert sample.shape == input_shape
        assert actual_log_prob.shape == (batch_size, 1)

    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=1.0, stop=2.0, step=0.25))
    def test_sample_multiple_and_compute_log_prob(self, mean, var):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)

        mu = np.ones(shape=input_shape) * mean
        ln_var = np.ones(shape=input_shape) * np.log(var)

        distribution = D.SquashedGaussian(mean=mu, ln_var=ln_var)
        num_samples = 10
        samples, log_probs = distribution.sample_multiple_and_compute_log_prob(
            num_samples=num_samples)
        # FIXME: if you enable clear_no_need_grad seems to compute something different
        # Do NOT use forward_all and no_need_grad flag at same time
        # NNabla's bug?
        nn.forward_all([samples, log_probs])

        assert np.alltrue(-1.0 <= samples.d)
        assert np.alltrue(samples.d <= 1.0)

        # Check the first sample independently
        x = np.arctanh(samples.d[:, 0, :], dtype=np.float64)
        assert x.shape == (batch_size, output_dim)
        gaussian_log_prob = -0.5 * np.log(2.0 * np.pi) - 0.5 * ln_var - \
            (x - mu) ** 2 / (2.0 * var)
        log_det_jacobian = np.log(1 - np.tanh(x) ** 2)
        expected = np.sum(gaussian_log_prob - log_det_jacobian,
                          axis=-1,
                          keepdims=True)
        actual = log_probs.d
        assert expected.shape == (batch_size, 1)
        assert np.allclose(expected, actual[:, 0, :], atol=1e-3, rtol=1e-2)

        # Check all the samples
        mu = np.reshape(mu, newshape=(batch_size, 1, output_dim))
        ln_var = np.reshape(ln_var, newshape=(batch_size, 1, output_dim))

        x = np.arctanh(samples.d, dtype=np.float64)
        gaussian_log_prob = -0.5 * np.log(2.0 * np.pi) - 0.5 * ln_var - \
            (x - mu) ** 2 / (2.0 * var)
        log_det_jacobian = np.log(1 - np.tanh(x) ** 2)
        expected = np.sum(gaussian_log_prob - log_det_jacobian,
                          axis=-1,
                          keepdims=True)
        actual = log_probs.d
        assert np.allclose(expected, actual, atol=1e-3, rtol=1e-2)

    def test_sample_multiple_and_compute_log_prob_shape(self):
        batch_size = 10
        output_dim = 10

        input_shape = (batch_size, output_dim)
        mean = np.zeros(shape=input_shape)
        sigma = np.ones(shape=input_shape) * 5.
        ln_var = np.log(sigma) * 2.

        distribution = D.SquashedGaussian(mean=mean, ln_var=ln_var)
        num_samples = 10
        samples, log_probs = distribution.sample_multiple_and_compute_log_prob(
            num_samples=num_samples)
        nn.forward_all([samples, log_probs])

        assert samples.shape == (batch_size, num_samples, output_dim)
        assert log_probs.shape == (batch_size, num_samples, 1)

    @pytest.mark.parametrize("x", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("mean", np.arange(start=-1.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("var", np.arange(start=1.0, stop=2.0, step=0.25))
    def test_log_prob_internal(self, x, mean, var):
        input_shape = (10, 10)
        dummy_mean = np.zeros(shape=input_shape)
        dummy_sigma = np.ones(shape=input_shape)
        dummy_ln_var = np.log(dummy_sigma) * 2.0

        distribution = D.SquashedGaussian(mean=dummy_mean, ln_var=dummy_ln_var)

        ln_var = np.log(var)
        gaussian_log_prob = -0.5 * \
            np.log(2.0 * np.pi) - 0.5 * ln_var - \
            (x - mean) ** 2 / (2.0 * var)
        log_det_jacobian = np.log(1 - np.tanh(x) ** 2)
        expected = np.sum(gaussian_log_prob -
                          log_det_jacobian, axis=-1, keepdims=True)

        x_var = nn.Variable((1, 1))
        x_var.d = x
        actual = distribution._log_prob_internal(
            x_var, mean=mean, var=var, ln_var=ln_var)
        actual.forward(clear_no_need_grad=True)
        actual = actual.d
        assert np.isclose(expected, actual)


if __name__ == "__main__":
    pytest.main()
