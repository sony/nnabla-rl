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

from typing import Callable

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.functions as RF
from nnabla_rl.models import (DiscreteQuantileDistributionFunction, DiscreteStateActionQuantileFunction,
                              DiscreteValueDistributionFunction)


class C51ValueDistributionFunction(DiscreteValueDistributionFunction):
    def all_probs(self, s: nn.Variable) -> nn.Variable:
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
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
                    h, n_outmaps=self._n_action * self._n_atom)
            h = NF.reshape(h, (-1, self._n_action, self._n_atom))
        assert h.shape == (batch_size, self._n_action, self._n_atom)
        return NF.softmax(h, axis=2)


class QRDQNQuantileDistributionFunction(DiscreteQuantileDistributionFunction):
    def all_quantiles(self, s: nn.Variable) -> nn.Variable:
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
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
                h = NPF.affine(h, n_outmaps=self._n_action * self._n_quantile)
            quantiles = NF.reshape(
                h, (-1, self._n_action, self._n_quantile))
        assert quantiles.shape == (batch_size, self._n_action, self._n_quantile)
        return quantiles


class IQNQuantileFunction(DiscreteStateActionQuantileFunction):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _embedding_dim: int

    def __init__(self, scope_name: str, n_action: int, embedding_dim: int, K: int,
                 risk_measure_function: Callable[[nn.Variable], nn.Variable]):
        super(IQNQuantileFunction, self).__init__(scope_name, n_action, K, risk_measure_function)
        self._embedding_dim = embedding_dim

    def all_quantile_values(self, s: nn.Variable, tau: nn.Variable) -> nn.Variable:
        encoded = self._encode(s, tau.shape[-1])
        embedding = self._compute_embedding(tau, encoded.shape[-1])

        assert embedding.shape == encoded.shape

        with nn.parameter_scope(self.scope_name):
            h = encoded * embedding
            with nn.parameter_scope("affine1"):
                h = NPF.affine(h, n_outmaps=512, base_axis=2)
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                return_samples = NPF.affine(h, n_outmaps=self._n_action, base_axis=2)
        assert return_samples.shape == (s.shape[0], tau.shape[-1], self._n_action)
        return return_samples

    def _encode(self, s: nn.Variable, n_sample: int) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, stride=(4, 4), kernel=(8, 8))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(s.shape[0], -1))
        encoded = RF.expand_dims(h, axis=1)
        encoded = RF.repeat(encoded, repeats=n_sample, axis=1)
        return encoded

    def _compute_embedding(self, tau: nn.Variable, dimension: int) -> nn.Variable:
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

    @property
    def _pi_i(self) -> nn.Variable:
        return np.pi * NF.arange(1, self._embedding_dim + 1)
