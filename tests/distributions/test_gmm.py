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

from unittest import mock

import numpy as np
import pytest
from scipy import stats

import nnabla as nn
from nnabla_rl.distributions.gmm import (GMM, NumpyGMM, compute_mean_and_covariance, compute_mixing_coefficient,
                                         compute_responsibility, inference_data_mean_and_covariance)


def _generate_dummy_params():
    num_classes = 2
    dims = 1
    means = np.array([[3.0], [5.0]])
    covariances = np.array([[[0.4]], [[0.1]]])
    mixing_coefficients = np.array([0.4, 0.6])
    return num_classes, dims, means, covariances, mixing_coefficients


class TestGMM():
    def test_nnabla_constructor(self):
        _, _, means, covariances, mixing_coefficients = _generate_dummy_params()
        nnabla_means = nn.Variable.from_numpy_array(means)
        nnabla_covariances = nn.Variable.from_numpy_array(covariances)
        nnabla_mixing_coefficients = nn.Variable.from_numpy_array(mixing_coefficients)
        with pytest.raises(NotImplementedError):
            GMM(nnabla_means, nnabla_covariances, nnabla_mixing_coefficients)

    def test_numpy_constructor(self):
        _, _, means, covariances, mixing_coefficients = _generate_dummy_params()
        distribution = GMM(means, covariances, mixing_coefficients)
        assert isinstance(distribution._delegate, NumpyGMM)


class TestNumpyGMM():
    def test_sample(self):
        _, dims, means, covariances, mixing_coefficients = _generate_dummy_params()
        gmm = NumpyGMM(means, covariances, mixing_coefficients)
        sample = gmm.sample()
        assert sample.shape == (dims, )

        with pytest.raises(NotImplementedError):
            gmm.sample(noise_clip=np.ones(means.shape[1:]))

    def test_numpy_log_prob(self):
        num_classes, dims, means, covariances, mixing_coefficients = _generate_dummy_params()
        data = np.random.randn(10, dims)

        gmm = NumpyGMM(means, covariances, mixing_coefficients)
        actual_log_probs = gmm.log_prob(data)

        probs = np.zeros((data.shape[0], num_classes))
        for k in range(num_classes):
            probs[:, k] = mixing_coefficients[k] * stats.multivariate_normal.pdf(data, means[k], covariances[k])

        assert np.allclose(actual_log_probs, np.log(probs))

    def test_numpy_compute_responsibility(self):
        num_classes, dims, means, covariances, mixing_coefficients = _generate_dummy_params()
        data = np.random.randn(10, dims)

        gmm = NumpyGMM(means, covariances, mixing_coefficients)

        _, actual_responsibility = compute_responsibility(data, gmm)

        probs = np.zeros((data.shape[0], num_classes))
        for k in range(num_classes):
            probs[:, k] = mixing_coefficients[k] * stats.multivariate_normal.pdf(data, means[k], covariances[k])
        denom = np.sum(probs, axis=1, keepdims=True)
        expected_responsibility = probs / denom

        assert np.allclose(actual_responsibility, expected_responsibility)

    def test_numpy_compute_mixing_coefficient(self):
        responsibility = np.random.randn(3, 1) + 2.0
        with mock.patch(
            'nnabla_rl.distributions.gmm.logsumexp', return_value=1.0
        ) as mock_logsumexp:
            result = compute_mixing_coefficient(responsibility)
            mock_logsumexp.assert_called_once()
            assert result == np.exp(1 - np.log(responsibility.shape[0]))

    def test_numpy_compute_mean_and_covariance(self):
        _, _, means, covariances, mixing_coefficients = _generate_dummy_params()
        gmm = NumpyGMM(means, covariances, mixing_coefficients)
        actual_mean, actual_var = compute_mean_and_covariance(gmm)

        assert np.allclose(actual_mean, np.array([4.2]))
        assert np.allclose(actual_var, np.array([[0.4 * 0.4 + 0.6 * 0.1 + 0.4 * 0.6 * (3 - 5) ** 2]]))

    def test_numpy_inference_data_mean_and_covariance(self):
        _, dims, means, covariances, mixing_coefficients = _generate_dummy_params()
        data = np.random.randn(10, dims)

        gmm = NumpyGMM(means, covariances, mixing_coefficients)

        with mock.patch(
            'nnabla_rl.distributions.gmm.compute_responsibility', return_value=(None, None)
        ) as mock_compute_responsibility:
            with mock.patch(
                'nnabla_rl.distributions.gmm.compute_mixing_coefficient', return_value=None
            ) as mock_compute_mixing_coefficient:
                with mock.patch(
                    'nnabla_rl.distributions.gmm.compute_mean_and_covariance',
                    return_value=(None, None),
                ) as mock_compute_mean_and_var:

                    inference_data_mean_and_covariance(data, gmm)
                    mock_compute_responsibility.assert_called_once()
                    mock_compute_mixing_coefficient.assert_called_once()
                    mock_compute_mean_and_var.assert_called_once()


if __name__ == '__main__':
    pytest.main()
