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


class TestSoftmax(object):
    def test_sample(self):
        z = np.array([[0, 0, 1000, 0],
                      [0, 1000, 0, 0],
                      [1000, 0, 0, 0],
                      [0, 0, 0, 1000]])

        batch_size = z.shape[0]
        distribution = D.Softmax(z=z)
        sampled = distribution.sample()

        sampled.forward()
        assert sampled.shape == (batch_size, 1)
        assert np.all(sampled.d == np.array([[2], [1], [0], [3]]))

    def test_log_prob(self):
        batch_size = 10
        action_num = 4
        z = np.random.normal(size=(batch_size, action_num))
        actions = np.array([[i % action_num] for i in range(batch_size)])

        distribution = D.Softmax(z=z)
        actual = distribution.log_prob(nn.Variable.from_numpy_array(actions))
        actual.forward()
        assert actual.shape == (batch_size, 1)

        probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        log_probabilities = np.log(probabilities)
        indices = np.reshape(actions, newshape=(batch_size, ))
        one_hot_action = self._to_one_hot_action(
            indices, action_num=action_num)
        expected = np.sum(log_probabilities * one_hot_action,
                          axis=1,
                          keepdims=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_entropy(self):
        batch_size = 10
        action_num = 4
        z = np.random.normal(size=(batch_size, action_num))
        distribution = D.Softmax(z=z)
        actual = distribution.entropy()
        actual.forward()
        assert actual.shape == (batch_size, 1)

        probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        expected = -np.sum(np.log(probabilities) *
                           probabilities, axis=1, keepdims=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    def test_kl_divergence(self):
        nn.clear_parameters()

        batch_size = 1
        z_p = np.array([[0.25, 0.95]])
        z_p_dist = self._softmax(z_p)
        distribution_p = D.Softmax(z=z_p)

        z_q = np.array([[0.5, 1.5]])
        z_q_dist = self._softmax(z_q)
        distribution_q = D.Softmax(z=z_q)

        actual = distribution_p.kl_divergence(distribution_q)
        actual.forward()

        assert actual.shape == (batch_size, 1)

        expected = z_p_dist[0, 0] * np.log(z_p_dist[0, 0] / z_q_dist[0, 0]) + \
            z_p_dist[0, 1] * np.log(z_p_dist[0, 1] / z_q_dist[0, 1])

        assert expected == pytest.approx(actual.d.flatten()[0], abs=1e-5)

    def _to_one_hot_action(self, a, action_num):
        action = a
        return np.eye(action_num, dtype=np.float32)[action]

    def _softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)


if __name__ == "__main__":
    pytest.main()
