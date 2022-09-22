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

import warnings
from typing import Union

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.distributions import Distribution, common_utils


class Gaussian(Distribution):
    '''
    Gaussian distribution

    :math:`\\mathcal{N}(\\mu,\\,\\sigma^{2})`

    Args:
        mean (nn.Variable): mean :math:`\\mu` of gaussian distribution.
        ln_var (nn.Variable): logarithm of the variance :math:`\\sigma^{2}`. (i.e. ln_var is :math:`\\log{\\sigma^{2}}`)
    '''

    _delegate: Union["NumpyGaussian", "NnablaGaussian"]

    def __init__(self, mean: Union[nn.Variable, np.ndarray], ln_var: Union[nn.Variable, np.ndarray]):
        super(Gaussian, self).__init__()
        if isinstance(mean, np.ndarray) and isinstance(ln_var, np.ndarray):
            warnings.warn(
                "Numpy ndarrays are given as mean and ln_var.\n"
                "From v0.12.0, if numpy.ndarray is given, "
                "all Gaussian class methods return numpy.ndarray not nnabla.Variable")
            self._delegate = NumpyGaussian(mean, ln_var)
        elif isinstance(mean, nn.Variable) and isinstance(ln_var, nn.Variable):
            self._delegate = NnablaGaussian(mean, ln_var)
        else:
            raise ValueError(
                f"Invalid type or a pair of types, mean type is {type(mean)} and ln type is {type(ln_var)}")

    @property
    def ndim(self):
        return self._delegate.ndim

    def sample(self, noise_clip=None):
        return self._delegate.sample(noise_clip)

    def sample_multiple(self, num_samples, noise_clip=None):
        return self._delegate.sample_multiple(num_samples, noise_clip)

    def sample_and_compute_log_prob(self, noise_clip=None):
        return self._delegate.sample_and_compute_log_prob(noise_clip)

    def sample_multiple_and_compute_log_prob(self, num_samples, noise_clip=None):
        return self._delegate.sample_multiple_and_compute_log_prob(num_samples, noise_clip)

    def choose_probable(self):
        return self._delegate.choose_probable()

    def mean(self):
        return self._delegate.mean()

    def var(self):
        return self._delegate.var()

    def log_prob(self, x):
        return self._delegate.log_prob(x)

    def entropy(self):
        return self._delegate.entropy()

    def kl_divergence(self, q):
        assert isinstance(q, Gaussian)
        return self._delegate.kl_divergence(q._delegate)


class NnablaGaussian(Distribution):
    _mean: nn.Variable
    _var: nn.Variable

    def __init__(self, mean: nn.Variable, ln_var: nn.Variable):
        super(Distribution, self).__init__()
        assert mean.shape == ln_var.shape
        self._mean = mean
        self._var = NF.exp(ln_var)
        self._ln_var = ln_var
        self._batch_size = mean.shape[0]
        self._data_dim = mean.shape[1:]
        self._ndim = mean.shape[-1]

    @property
    def ndim(self):
        return self._ndim

    def sample(self, noise_clip=None):
        return RF.sample_gaussian(self._mean,
                                  self._ln_var,
                                  noise_clip=noise_clip)

    def sample_multiple(self, num_samples, noise_clip=None):
        return RF.sample_gaussian_multiple(self._mean,
                                           self._ln_var,
                                           num_samples,
                                           noise_clip=noise_clip)

    def sample_and_compute_log_prob(self, noise_clip=None):
        x = RF.sample_gaussian(mean=self._mean,
                               ln_var=self._ln_var,
                               noise_clip=noise_clip)
        return x, self.log_prob(x)

    def sample_multiple_and_compute_log_prob(self, num_samples, noise_clip=None):
        x = RF.sample_gaussian_multiple(self._mean,
                                        self._ln_var,
                                        num_samples,
                                        noise_clip=noise_clip)
        mean = RF.expand_dims(self._mean, axis=1)
        var = RF.expand_dims(self._var, axis=1)
        ln_var = RF.expand_dims(self._ln_var, axis=1)

        assert mean.shape == (self._batch_size, 1, ) + self._data_dim
        assert var.shape == mean.shape
        assert ln_var.shape == mean.shape

        return x, common_utils.gaussian_log_prob(x, mean, var, ln_var)

    def choose_probable(self):
        return self._mean

    def mean(self):
        return self._mean

    def var(self):
        return self._var

    def log_prob(self, x):
        return common_utils.gaussian_log_prob(x, self._mean, self._var, self._ln_var)

    def entropy(self):
        return NF.sum(0.5 + 0.5 * np.log(2.0 * np.pi) + 0.5 * self._ln_var, axis=1, keepdims=True)

    def kl_divergence(self, q):
        assert isinstance(q, NnablaGaussian)
        p = self
        return 0.5 * NF.sum(q._ln_var - p._ln_var + (p._var + (p._mean - q._mean) ** 2.0) / q._var - 1,
                            axis=1,
                            keepdims=True)


class NumpyGaussian(Distribution):
    _mean: np.ndarray
    _var: np.ndarray

    def __init__(self, mean: np.ndarray, ln_var: np.ndarray) -> None:
        super(Distribution, self).__init__()
        self._dim = mean.shape[0]
        assert (self._dim, ) == mean.shape
        assert (self._dim, self._dim) == ln_var.shape
        self._mean = mean
        self._var = np.exp(ln_var)
        self._inv_var = np.linalg.inv(self._var)

    def log_prob(self, x):
        log_det_term = np.log(np.linalg.det(2.0 * np.pi * self._var))
        diff = self._mean - x
        quadratic_term = diff.T.dot(self._inv_var).dot(diff)
        return -0.5 * (log_det_term + quadratic_term)

    def mean(self):
        return self._mean

    def var(self):
        return self._var

    def sample(self, noise_clip=None):
        if noise_clip is not None:
            raise NotImplementedError
        return np.random.multivariate_normal(self._mean, self._var)

    def sample_and_compute_log_prob(self, noise_clip=None):
        raise NotImplementedError

    def sample_multiple(self, num_samples, noise_clip=None):
        raise NotImplementedError

    def sample_multiple_and_compute_log_prob(self, num_samples, noise_clip=None):
        raise NotImplementedError

    def kl_divergence(self, q: 'Distribution'):
        if not isinstance(q, NumpyGaussian):
            raise NotImplementedError

        p_mean = self._mean
        p_var = self._var

        q_mean = q.mean()
        q_var = q.var()
        q_var_inv = np.linalg.inv(q.var())

        trace_term = np.trace(q_var_inv.dot(p_var))
        diff = q_mean - p_mean
        quadratic_term = diff.T.dot(q_var_inv).dot(diff)
        dimension = self._dim
        log_det_term = np.log(np.linalg.det(q_var)) - np.log(np.linalg.det(p_var))
        return 0.5 * (trace_term + quadratic_term - dimension + log_det_term)
