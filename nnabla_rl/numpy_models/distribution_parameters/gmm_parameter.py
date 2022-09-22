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

from nnabla_rl.numpy_models.distribution_parameter import DistributionParameter


class GMMParameter(DistributionParameter):
    '''Gaussian Mixture Model Parameter'''

    _means: np.ndarray
    _covarinces: np.ndarray
    _mixing_coefficients: np.ndarray

    def __init__(self, means: np.ndarray,
                 covariances: np.ndarray,
                 mixing_coefficients: np.ndarray) -> None:
        super().__init__()
        self._num_classes, self._dim = means.shape
        assert (self._num_classes, self._dim, self._dim) == covariances.shape
        assert (self._num_classes, ) == mixing_coefficients.shape
        self._means = means
        self._covariances = covariances
        self._mixing_coefficients = mixing_coefficients

    @staticmethod
    def from_data(data: np.ndarray, num_classes: int) -> 'GMMParameter':
        '''create GMM from data by random class assignnment

        Args:
            data (np.ndarray): data, shape (num_data, dim)
            num_classes (int): number of classes

        Returns:
            GMMParameter: gaussian mixture model parameter
        '''
        num_data, dim = data.shape

        mixing_coefficients = 1.0 / num_classes * np.ones(num_classes)
        means = np.zeros((num_classes, dim))
        covariances = np.zeros((num_classes, dim, dim))

        # assign random class to data
        assigned_class = np.random.randint(0, num_classes, size=num_data)

        # compute each class's mean and value
        for i in range(num_classes):
            class_idx = np.where(assigned_class == i)[0]
            mean = np.mean(data[class_idx, :], axis=0)
            diff = (data[class_idx, :] - mean).T
            sigma = (1.0 / num_classes) * (diff.dot(diff.T))
            means[i, :] = mean
            covariances[i, :, :] = sigma + np.eye(dim) * 1e-8  # add small value to keep positive-definiy

        return GMMParameter(means, covariances, mixing_coefficients)

    def update_parameter(self,  # type: ignore
                         new_means: np.ndarray,
                         new_covariances: np.ndarray,
                         new_mixing_coefficients: np.ndarray) -> None:
        self._means = new_means
        self._covariances = new_covariances
        self._mixing_coefficients = new_mixing_coefficients
