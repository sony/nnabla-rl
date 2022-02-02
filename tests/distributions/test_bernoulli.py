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
import nnabla_rl.distributions as D


class TestBernoulli(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_sample(self):
        z = np.array([[1000.0], [-1000.0], [-1000.0], [1000.0]])

        batch_size = z.shape[0]
        distribution = D.Bernoulli(z=z)
        sampled = distribution.sample()

        sampled.forward()
        assert sampled.shape == (batch_size, 1)
        assert np.all(sampled.d == np.array([[1], [0], [0], [1]]))

    def test_sample_multi_dimensional(self):
        z = np.array([[[1000.0], [-1000.0], [-1000.0], [1000.0]],
                      [[1000.0], [-1000.0], [1000.0], [-1000.0]]])
        assert z.shape == (2, 4, 1)
        batch_size = z.shape[0]
        category_size = z.shape[1]
        distribution = D.Bernoulli(z=z)
        sampled = distribution.sample()

        sampled.forward()
        assert sampled.shape == (batch_size, category_size, 1)
        assert np.all(sampled.d == np.array([[[1], [0], [0], [1]], [[1], [0], [1], [0]]]))

    def test_log_prob(self):
        batch_size = 10
        z = np.random.normal(size=(batch_size, 1))
        classes = np.random.randint(2, size=(batch_size, 1))

        distribution = D.Bernoulli(z=z)
        actual = distribution.log_prob(nn.Variable.from_numpy_array(classes))
        actual.forward()
        assert actual.shape == (batch_size, 1)

        p = self._sigmoid(z)
        expected = np.where(classes == 1, np.log(p), np.log(1-p))
        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_log_prob_multi_dimensional(self):
        batch_size = 10
        category_num = 5
        z = np.random.normal(size=(batch_size, category_num, 1))
        classes = np.random.randint(2, size=(batch_size, category_num, 1))

        distribution = D.Bernoulli(z=z)
        actual = distribution.log_prob(nn.Variable.from_numpy_array(classes))
        actual.forward()
        assert actual.shape == (batch_size, category_num, 1)

        p = self._sigmoid(z)
        expected = np.where(classes == 1, np.log(p), np.log(1-p))
        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_entropy(self):
        batch_size = 10
        z = np.random.normal(size=(batch_size, 1))
        distribution = D.Bernoulli(z=z)
        actual = distribution.entropy()
        actual.forward()
        assert actual.shape == (batch_size, 1)

        p = self._sigmoid(z)
        probabilities = np.concatenate((p, 1-p), axis=-1)
        expected = -np.sum(np.log(probabilities) * probabilities, axis=1, keepdims=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_entropy_multi_dimensional(self):
        batch_size = 10
        category_num = 3
        z = np.random.normal(size=(batch_size, category_num, 1))
        distribution = D.Bernoulli(z=z)
        actual = distribution.entropy()
        actual.forward()
        assert actual.shape == (batch_size, category_num, 1)

        p = self._sigmoid(z)
        probabilities = np.concatenate((p, 1-p), axis=-1)
        expected = -np.sum(np.log(probabilities) * probabilities, axis=len(z.shape) - 1, keepdims=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_kl_divergence(self):
        batch_size = 10
        z_p = np.random.normal(size=(batch_size, 1))
        z_p_p = self._sigmoid(z_p)
        z_p_dist = np.concatenate((z_p_p, 1-z_p_p), axis=-1)
        distribution_p = D.Bernoulli(z=z_p)

        z_q = np.random.normal(size=(batch_size, 1))
        z_q_p = self._sigmoid(z_q)
        z_q_dist = np.concatenate((z_q_p, 1-z_q_p), axis=-1)
        distribution_q = D.Bernoulli(z=z_q)

        actual = distribution_p.kl_divergence(distribution_q)
        actual.forward()

        assert actual.shape == (batch_size, 1)

        expected = np.sum(z_p_dist * np.log(z_p_dist) - z_p_dist * np.log(z_q_dist), axis=-1, keepdims=True)
        assert expected.shape == (batch_size, 1)
        assert np.allclose(actual.d, expected, atol=1e-5)

    def test_kl_divergence_multi_dimensional(self):
        batch_size = 10
        category_num = 3
        z_p = np.random.normal(size=(batch_size, category_num, 1))
        z_p_p = self._sigmoid(z_p)
        z_p_dist = np.concatenate((z_p_p, 1-z_p_p), axis=-1)
        distribution_p = D.Bernoulli(z=z_p)

        z_q = np.random.normal(size=(batch_size, category_num, 1))
        z_q_p = self._sigmoid(z_q)
        z_q_dist = np.concatenate((z_q_p, 1-z_q_p), axis=-1)
        distribution_q = D.Bernoulli(z=z_q)

        actual = distribution_p.kl_divergence(distribution_q)
        actual.forward()

        assert actual.shape == (batch_size, category_num, 1)

        expected = np.sum(z_p_dist * np.log(z_p_dist) - z_p_dist * np.log(z_q_dist), axis=-1, keepdims=True)
        assert expected.shape == (batch_size, category_num, 1)
        assert np.allclose(actual.d, expected, atol=1e-5)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    pytest.main()
