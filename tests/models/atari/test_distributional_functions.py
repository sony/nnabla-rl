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
from nnabla_rl.models.atari.distributional_functions import (C51ValueDistributionFunction, IQNQuantileFunction,
                                                             QRDQNQuantileDistributionFunction)


def risk_measure_function(tau):
    return tau


class TestC51ValueDistributionFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        n_action = 4
        n_atom = 10
        scope_name = "test"
        v_min = 0
        v_max = 10
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             n_action=n_action,
                                             n_atom=n_atom,
                                             v_max=v_max,
                                             v_min=v_min)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_atom = 10
        scope_name = "test"
        v_min = 0
        v_max = 10
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             n_action=n_action,
                                             n_atom=n_atom,
                                             v_max=v_max,
                                             v_min=v_min)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        model.all_probs(input_state)
        assert len(model.get_parameters()) == 10

    def test_probabilities(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_atom = 10
        scope_name = "test"
        v_min = 0
        v_max = 10
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             n_action=n_action,
                                             n_atom=n_atom,
                                             v_max=v_max,
                                             v_min=v_min)

        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        val = model.all_probs(input_state)
        val.forward()

        assert val.shape == (1, n_action, n_atom)
        assert np.alltrue(0.0 <= val.d) and np.alltrue(val.d <= 1.0)


class TestQRDQNQuantileDistributionFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        n_action = 4
        n_quantile = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  n_action=n_action,
                                                  n_quantile=n_quantile)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_quantile = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  n_action=n_action,
                                                  n_quantile=n_quantile)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        model.all_quantiles(input_state)
        assert len(model.get_parameters()) == 10

    def test_quantiles(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_quantile = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  n_action=n_action,
                                                  n_quantile=n_quantile)

        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        val = model.all_quantiles(input_state)
        val.forward()

        assert val.shape == (1, n_action, n_quantile)


class TestIQNQuantileFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        n_action = 4
        embedding_dim = 10
        scope_name = "test"
        K = 10
        model = IQNQuantileFunction(scope_name=scope_name,
                                    n_action=n_action,
                                    embedding_dim=embedding_dim,
                                    K=K,
                                    risk_measure_function=risk_measure_function)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 10
        scope_name = "test"
        K = 10
        model = IQNQuantileFunction(scope_name=scope_name,
                                    n_action=n_action,
                                    embedding_dim=embedding_dim,
                                    K=K,
                                    risk_measure_function=risk_measure_function)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        tau = nn.Variable.from_numpy_array(np.random.rand(1, 10))
        model.all_quantile_values(input_state, tau)
        assert len(model.get_parameters()) == 12

    def test_quantile_values(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        K = 10
        model = IQNQuantileFunction(scope_name=scope_name,
                                    n_action=n_action,
                                    embedding_dim=embedding_dim,
                                    K=K,
                                    risk_measure_function=risk_measure_function)

        # Initialize parameters
        n_sample = 5
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        tau = nn.Variable.from_numpy_array(np.random.rand(1, n_sample))
        return_samples = model.all_quantile_values(input_state, tau)
        return_samples.forward()

        assert return_samples.shape == (1, n_sample, n_action)

    def test_encode(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        K = 10
        model = IQNQuantileFunction(scope_name=scope_name,
                                    n_action=n_action,
                                    embedding_dim=embedding_dim,
                                    K=K,
                                    risk_measure_function=risk_measure_function)

        # Initialize parameters
        n_sample = 5
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        encoded = model._encode(input_state, n_sample=n_sample)
        encoded.forward()
        encoded = encoded.d

        assert encoded.shape == (1, n_sample, 3136)
        assert np.alltrue(encoded[:, 1:, :] == encoded[:, 0, :])
        print('encoded: ', encoded)

    def test_compute_embeddings(self):
        nn.clear_parameters()

        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        K = 10
        model = IQNQuantileFunction(scope_name=scope_name,
                                    n_action=n_action,
                                    embedding_dim=embedding_dim,
                                    K=K,
                                    risk_measure_function=risk_measure_function)

        # Initialize parameters
        n_sample = 5
        encode_dim = 3
        tau = np.random.rand(1, n_sample)
        tau_var = nn.Variable.from_numpy_array(tau)
        embedding = model._compute_embedding(tau_var, dimension=encode_dim)

        params = model.get_parameters()
        for key, param in params.items():
            if 'embedding' not in key:
                continue
            param.d = np.ones(param.shape)
        embedding.forward()
        actual = embedding.d

        expected = []
        for t in tau[0]:
            for _ in range(encode_dim):
                embedding = np.sum(
                    [np.cos(np.pi * (i + 1) * t) for i in range(embedding_dim)])
                embedding += 1  # Add bias
                embedding = np.maximum(0.0, embedding)
                expected.append(embedding)

        expected = np.array(expected).reshape((1, n_sample, encode_dim))

        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
