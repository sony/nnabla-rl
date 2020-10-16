import pytest
from unittest import mock

import numpy as np

import nnabla as nn

from nnabla_rl.models.atari.distributional_functions import C51ValueDistributionFunction, QRDQNQuantileDistributionFunction, IQNQuantileFunction


class TestC51ValueDistributionFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_atoms = 10
        scope_name = "test"
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             state_shape=state_shape,
                                             num_actions=n_action,
                                             num_atoms=n_atoms)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_atoms = 10
        scope_name = "test"
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             state_shape=state_shape,
                                             num_actions=n_action,
                                             num_atoms=n_atoms)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        model.probabilities(input_state)
        assert len(model.get_parameters()) == 10

    def test_probabilities(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_atoms = 10
        scope_name = "test"
        model = C51ValueDistributionFunction(scope_name=scope_name,
                                             state_shape=state_shape,
                                             num_actions=n_action,
                                             num_atoms=n_atoms)

        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        val = model.probabilities(input_state)
        val.forward()

        assert val.shape == (1, n_action, n_atoms)
        assert np.alltrue(0.0 <= val.d) and np.alltrue(val.d <= 1.0)


class TestQRDQNQuantileDistributionFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_quantiles = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  state_shape=state_shape,
                                                  num_actions=n_action,
                                                  num_quantiles=n_quantiles)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_quantiles = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  state_shape=state_shape,
                                                  num_actions=n_action,
                                                  num_quantiles=n_quantiles)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        model.quantiles(input_state)
        assert len(model.get_parameters()) == 10

    def test_quantiles(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        n_quantiles = 10
        scope_name = "test"
        model = QRDQNQuantileDistributionFunction(scope_name=scope_name,
                                                  state_shape=state_shape,
                                                  num_actions=n_action,
                                                  num_quantiles=n_quantiles)

        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        val = model.quantiles(input_state)
        val.forward()

        assert val.shape == (1, n_action, n_quantiles)


class TestIQNQuantileFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 10
        scope_name = "test"
        model = IQNQuantileFunction(scope_name=scope_name,
                                    state_shape=state_shape,
                                    num_actions=n_action,
                                    embedding_dim=embedding_dim)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 10
        scope_name = "test"
        model = IQNQuantileFunction(scope_name=scope_name,
                                    state_shape=state_shape,
                                    num_actions=n_action,
                                    embedding_dim=embedding_dim)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        tau = nn.Variable.from_numpy_array(
            np.random.rand(1, 10))
        model.quantiles(input_state, tau)
        assert len(model.get_parameters()) == 12

    def test_quantiles(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        model = IQNQuantileFunction(scope_name=scope_name,
                                    state_shape=state_shape,
                                    num_actions=n_action,
                                    embedding_dim=embedding_dim)

        # Initialize parameters
        num_samples = 5
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        tau = nn.Variable.from_numpy_array(np.random.rand(1, num_samples))
        quantiles = model.quantiles(input_state, tau)
        quantiles.forward()

        assert quantiles.shape == (1, num_samples, n_action)

    def test_encode(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        model = IQNQuantileFunction(scope_name=scope_name,
                                    state_shape=state_shape,
                                    num_actions=n_action,
                                    embedding_dim=embedding_dim)

        # Initialize parameters
        num_samples = 5
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        encoded = model._encode(input_state, num_samples=num_samples)
        encoded.forward()
        encoded = encoded.d

        assert encoded.shape == (1, num_samples, 3136)
        assert np.alltrue(encoded[:, 1:, :] == encoded[:, 0, :])
        print('encoded: ', encoded)

    def test_compute_embeddings(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        embedding_dim = 64
        scope_name = "test"
        model = IQNQuantileFunction(scope_name=scope_name,
                                    state_shape=state_shape,
                                    num_actions=n_action,
                                    embedding_dim=embedding_dim)

        # Initialize parameters
        num_samples = 5
        encode_dim = 3
        tau = np.random.rand(1, num_samples)
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
                    [np.cos(np.pi * i * t) for i in range(embedding_dim)])
                embedding += 1  # Add bias
                embedding = np.maximum(0.0, embedding)
                expected.append(embedding)

        expected = np.array(expected).reshape((1, num_samples, encode_dim))

        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
