import nnabla as nn

import nnabla.functions as NF
import nnabla.parametric_functions as NPF

from nnabla_rl.models import ValueDistributionFunction, QuantileDistributionFunction, StateActionQuantileFunction
import nnabla_rl.functions as RF
import numpy as np


class C51ValueDistributionFunction(ValueDistributionFunction):
    def __init__(self, scope_name, state_shape, num_actions, num_atoms):
        super(C51ValueDistributionFunction, self).__init__(scope_name)
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._num_atoms = num_atoms

    def probabilities(self, s):
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            assert s.shape[1:] == self._state_shape
            batch_size = s.shape[0]

            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, stride=(4, 4), kernel=(8, 8))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
            with nn.parameter_scope("affine1"):
                h = NPF.affine(h, n_outmaps=512)
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                h = NPF.affine(
                    h, n_outmaps=self._num_actions * self._num_atoms)
            h = NF.reshape(h, (-1, self._num_actions, self._num_atoms))
        assert h.shape == (batch_size, self._num_actions, self._num_atoms)
        return NF.softmax(h, axis=2)


class QRDQNQuantileDistributionFunction(QuantileDistributionFunction):
    def __init__(self, scope_name, state_shape, num_actions, num_quantiles):
        super(QRDQNQuantileDistributionFunction, self).__init__(scope_name)
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._num_quantiles = num_quantiles

    def quantiles(self, s):
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            assert s.shape[1:] == self._state_shape
            batch_size = s.shape[0]

            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, stride=(4, 4), kernel=(8, 8))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
            with nn.parameter_scope("affine1"):
                h = NPF.affine(h, n_outmaps=512)
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                h = NPF.affine(h, n_outmaps=self._num_actions * self._num_quantiles)
            quantiles = NF.reshape(
                h, (-1, self._num_actions, self._num_quantiles))
        assert quantiles.shape == (
            batch_size, self._num_actions, self._num_quantiles)
        return quantiles


class IQNQuantileFunction(StateActionQuantileFunction):
    def __init__(self, scope_name, state_shape, num_actions, embedding_dim):
        super(IQNQuantileFunction, self).__init__(scope_name)
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._embedding_dim = embedding_dim

        self._pi_i = np.pi * nn.Variable.from_numpy_array(
            np.array([i for i in range(0, embedding_dim)]))

    def quantiles(self, s, tau):
        batch_size = s.shape[0]

        encoded = self._encode(s, tau.shape[-1])
        embedding = self._compute_embedding(tau, encoded.shape[-1])

        assert embedding.shape == encoded.shape

        with nn.parameter_scope(self.scope_name):
            h = encoded * embedding
            with nn.parameter_scope("affine1"):
                h = NPF.affine(h, n_outmaps=512, base_axis=2)
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                quantile = NPF.affine(
                    h, n_outmaps=self._num_actions, base_axis=2)
        assert quantile.shape == (
            batch_size, tau.shape[-1], self._num_actions)
        return quantile

    def _encode(self, s, num_samples):
        with nn.parameter_scope(self.scope_name):
            assert s.shape[1:] == self._state_shape
            batch_size = s.shape[0]

            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, stride=(4, 4), kernel=(8, 8))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
        encoded = RF.expand_dims(h, axis=1)
        encoded = RF.repeat(encoded, repeats=num_samples, axis=1)
        return encoded

    def _compute_embedding(self, tau, dimension):
        with nn.parameter_scope(self.scope_name):
            batch_size = tau.shape[0]
            sample_size = tau.shape[1]

            tau = RF.expand_dims(tau, axis=2)
            tau = RF.repeat(tau, repeats=self._embedding_dim, axis=2)
            assert tau.shape == (batch_size, sample_size, self._embedding_dim)

            pi_i = NF.reshape(self._pi_i, (1, 1, self._embedding_dim))
            pi_i = RF.repeat(pi_i, repeats=sample_size, axis=1)
            pi_i = RF.repeat(pi_i, repeats=batch_size, axis=0)

            assert tau.shape == pi_i.shape

            h = NF.cos(pi_i * tau)
            with nn.parameter_scope("embedding1"):
                h = NPF.affine(h, n_outmaps=dimension, base_axis=2)
            embedding = NF.relu(x=h)
        assert embedding.shape == (batch_size, sample_size, dimension)
        return embedding
