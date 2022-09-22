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

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import numpy as np
from scipy import linalg

import nnabla as nn
from nnabla_rl.distributions.distribution import Distribution

if TYPE_CHECKING:
    from nnabla_rl.numpy_models.distribution_parameters.gmm_parameter import GMMParameter


class GMM(Distribution):
    '''
    Gaussian Mixture Model distribution

    :math:`\\Sum_{k=1}^{N} \\pi_k \\mathcal{N}(\\mu,\\,\\Sigma)`

    Args:
        means (Union[nn.Variable, np.ndarray]): means of each gaussian distribution :math:`\\mu_k`.
        covariances (Union[nn.Variable, np.ndarray]): covariances of each gaussian distribution. :math:`\\Sigma_k`.
        mixing_coefficients (Union[nn.Variable, np.ndarray]):
            mixing coefficients of each gaussian distribution. :math:`\\pi_k`.
    '''

    def __init__(self,
                 means: Union[nn.Variable, np.ndarray],
                 covariances: Union[nn.Variable, np.ndarray],
                 mixing_coefficients: Union[nn.Variable, np.ndarray]):
        super(GMM, self).__init__()
        if (
            isinstance(means, np.ndarray) and
            isinstance(covariances, np.ndarray) and
            isinstance(mixing_coefficients, np.ndarray)
        ):
            self._delegate = NumpyGMM(means, covariances, mixing_coefficients)
        elif (
            isinstance(means, nn.Variable) and
            isinstance(covariances, nn.Variable) and
            isinstance(mixing_coefficients, nn.Variable)
        ):
            raise NotImplementedError
        else:
            raise ValueError(
                "Invalid type or a set of types.\n"
                f"means type is {type(means)}, "
                f"covariances type is {type(covariances)} and"
                f"mixing_coefficients type is {type(mixing_coefficients)}")

    def sample(self, noise_clip=None):
        raise self._delegate.sample(noise_clip)

    @property
    def num_classes(self):
        return self._delegate.num_classes

    def log_prob(self, x):
        return self._delegate.log_prob(x)

    def mean(self):
        return self._delegate.mean()

    def covariance(self):
        return self._delegate.covariance()


class NumpyGMM(Distribution):
    _means: np.ndarray
    _covariances: np.ndarray
    _mixing_coefficients: np.ndarray

    def __init__(self, means: np.ndarray, covariances: np.ndarray, mixing_coefficients: np.ndarray) -> None:
        super(NumpyGMM, self).__init__()
        self._num_classes, self._dim = means.shape
        assert (self._num_classes, self._dim, self._dim) == covariances.shape
        assert (self._num_classes, ) == mixing_coefficients.shape
        self._means = means  # shape (num_classes, dim)
        self._covariances = covariances  # shape (num_classes, dim, dim)
        self._mixing_coefficients = mixing_coefficients  # shape(num_classes, )

    @staticmethod
    def from_gmm_parameter(parameter: 'GMMParameter') -> 'NumpyGMM':
        return NumpyGMM(parameter._means, parameter._covariances, parameter._mixing_coefficients)

    def sample(self, noise_clip=None):
        if noise_clip is not None:
            raise NotImplementedError
        sample_class = np.random.choice(self._num_classes, p=self._mixing_coefficients)
        return np.random.multivariate_normal(self._means[sample_class], self._covariances[sample_class])

    @property
    def num_classes(self):
        return self._num_classes

    def log_prob(self, x):
        '''compute log observation probabilities of each data under current parameters

        Args:
            x (np.ndarray): data, shape(num_data, dim)

        Returns:
            np.ndarray: log observation probabilities of each data
        '''
        num_samples, dim = x.shape
        assert self._dim == dim
        # compute constant part of multiple gaussian log prob
        log_probs = -0.5 * np.ones((num_samples, self._num_classes)) * self._dim * np.log(2 * np.pi)

        for i in range(self._num_classes):
            mean, covs = self._means[i], self._covariances[i]

            # compute determinant of cov marix with cholesky decomposition
            cholesky_decomposed_cov = linalg.cholesky(covs, lower=True)
            log_probs[:, i] -= np.sum(np.log(np.diag(cholesky_decomposed_cov)))

            # compute quadratic form value without solving inverse matrix
            diff = (x - mean).T  # shape (dim, num_data)
            diff_inv_cholesky_decomposed_cov = linalg.solve_triangular(cholesky_decomposed_cov, diff, lower=True)
            log_probs[:, i] -= 0.5 * np.sum(diff_inv_cholesky_decomposed_cov ** 2, axis=0)

        log_probs += np.log(self._mixing_coefficients)
        return log_probs

    def mean(self):
        mean, _ = compute_mean_and_covariance(self)
        return mean

    def covariance(self):
        _, covariance = compute_mean_and_covariance(self)
        return covariance


def compute_responsibility(data: np.ndarray, distribution: NumpyGMM) -> Tuple[np.ndarray, np.ndarray]:
    '''compute responsibility under current parameters

    Args:
        data (np.ndarray): data, shape(num_data, dim)
        distribution (NumpyGMM): distribution

    Returns:
        Tuple[np.ndarray, np.ndarray]: observation probability, responsibility
    '''
    log_probs = distribution.log_prob(data)
    log_responsibility = log_probs - logsumexp(log_probs, axis=1, keepdims=True)  # shape (num_data, num_classes)

    return np.exp(log_probs), np.exp(log_responsibility)


def compute_mean_and_covariance(distribution: NumpyGMM,
                                mixing_coefficients: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''compute mean and covariance of gmm

    Args:
        distribution (NumpyGMM): distribution
        mixing_coefficients (Optional[np.ndarray]): mixing coefficient, if not given, use current mixing coefficient

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean and var of gmm under current parameters
    '''
    if mixing_coefficients is None:
        # sum of all class's mean, shape (dim, )
        mean = np.sum(distribution._means *
                      distribution._mixing_coefficients[:, np.newaxis], axis=0)
    else:
        # sum of all class's mean, shape (dim, )
        mean = np.sum(distribution._means * mixing_coefficients[:, np.newaxis], axis=0)

    # diff.shape = (num_classes, dim)
    diff = distribution._means - mean[np.newaxis, :]
    # diff.shape = (num_classes, dim, dim)
    diff = distribution._means[:, np.newaxis] * diff[:, :, np.newaxis]
    # mixing_coefficients.shape = (num_classes, 1, 1)
    mixing_coefficients = distribution._mixing_coefficients[:, np.newaxis, np.newaxis]
    # covariance.shape = (dim, dim)
    covariance = np.sum((distribution._covariances + diff) * mixing_coefficients, axis=0)
    return mean, covariance


def compute_mixing_coefficient(responsibility: np.ndarray) -> np.ndarray:
    '''compute mixing coefficient from the given responsibility

    Args:
        responsibility (np.ndarray): responsibility, shape(num_data, num_classes)

    Returns:
        np.ndarray: mixing_coefficients
    '''
    num_data, _ = responsibility.shape
    # shape (num_classes, )
    log_mixing_coefficients = logsumexp(np.log(responsibility), axis=0, keepdims=True) - np.log(num_data)
    return cast(np.ndarray, np.exp(log_mixing_coefficients.flatten()))


def inference_data_mean_and_covariance(data: np.ndarray, distribution: NumpyGMM) -> Tuple[np.ndarray, np.ndarray]:
    '''inference mean and covariance of given data

    Args:
        data (np.ndarray): data, shape(num_data, dim)
        distribution (NumpyGMM): distribution

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean and var of data
    '''
    _, responsibility = compute_responsibility(data, distribution)
    mixing_coefficients = compute_mixing_coefficient(responsibility)
    mean, covariance = compute_mean_and_covariance(distribution, mixing_coefficients)
    return mean, covariance


def logsumexp(x: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    r'''compute log sum exp value of the input

    .. math::
        y = \log (\Sigma (\exp (input)))

    Args:
        x (np.ndarray): input
        axis (int): base axis of log sum operation
        keepdims (bool): flag whether the reduced axes are kept as a dimension with 1 element

    Returns:
        np.ndarray: log sum exp value of the input
    '''
    max_x = np.max(x, axis=axis, keepdims=True)
    max_x[max_x == -float('inf')] = 0.
    if keepdims:
        return cast(np.ndarray, np.log(np.sum(np.exp(x-max_x), axis=axis, keepdims=keepdims)) + max_x)
    else:
        return cast(np.ndarray, np.log(np.sum(np.exp(x-max_x),
                                              axis=axis, keepdims=keepdims)) + np.squeeze(max_x, axis=axis))
