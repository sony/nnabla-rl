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

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

import nnabla as nn


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, noise_clip: Optional[Tuple[float, float]] = None) -> Union[nn.Variable, np.ndarray]:
        '''
        Sample a value from the distribution.
        If noise_clip is specified, the sampled value will be clipped in the given range.
        Applicability of noise_clip depends on underlying implementation.

        Args:
            noise_clip(Tuple[float, float], optional):
                float tuple of size 2 which contains the min and max value of the noise.

        Returns:
            Union[nn.Variable, np.ndarray]: Sampled value
        '''
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        '''
        The number of dimensions of the distribution
        '''
        raise NotImplementedError

    def sample_multiple(self, num_samples: int, noise_clip: Optional[Tuple[float, float]] = None
                        ) -> Union[nn.Variable, np.ndarray]:
        '''
        Sample mutiple value from the distribution
        New axis will be added between the first and second axis.
        Thefore, the returned value shape for mean and variance with shape (batch_size, data_shape)
        will be changed to (batch_size, num_samples, data_shape)

        If noise_clip is specified, sampled values will be clipped in the given range.
        Applicability of noise_clip depends on underlying implementation.

        Args:
            num_samples(int): number of samples per batch
            noise_clip(Tuple[float, float], optional):
                float tuple of size 2 which contains the min and max value of the noise.

        Returns:
            Union[nn.Variable, np.ndarray]: Sampled value.
        '''
        raise NotImplementedError

    def choose_probable(self) -> Union[nn.Variable, np.ndarray]:
        '''
        Compute the most probable action of the distribution

        Returns:
            Union[nn.Variable, np.ndarray]: Probable action of the distribution
        '''
        raise NotImplementedError

    def mean(self) -> Union[nn.Variable, np.ndarray]:
        '''
        Compute the mean of the distribution (if exist)

        Returns:
            Union[nn.Variable, np.ndarray]: mean of the distribution

        Raises:
             NotImplementedError: The distribution does not have mean
        '''
        raise NotImplementedError

    def log_prob(self, x: Union[nn.Variable, np.ndarray]) -> Union[nn.Variable, np.ndarray]:
        '''
        Compute the log probability of given input

        Args:
            x (Union[nn.Variable, np.ndarray]): Target value to compute the log probability

        Returns:
            Union[nn.Variable, np.ndarray]: Log probability of given input
        '''
        raise NotImplementedError

    def sample_and_compute_log_prob(self, noise_clip: Optional[Tuple[float, float]] = None) \
            -> Union[Tuple[nn.Variable, nn.Variable], Tuple[np.ndarray, np.ndarray]]:
        '''
        Sample a value from the distribution and compute its log probability.

        Args:
            noise_clip(Tuple[float, float], optional):
                float tuple of size 2 which contains the min and max value of the noise.

        Returns:
            Union[Tuple[nn.Variable, nn.Variable], Tuple[np.ndarray, np.ndarray]]: Sampled value and its log probabilty
        '''
        raise NotImplementedError

    def entropy(self) -> Union[nn.Variable, np.ndarray]:
        '''
        Compute the entropy of the distribution

        Returns:
            Union[nn.Variable, np.ndarray]: Entropy of the distribution
        '''
        raise NotImplementedError

    def kl_divergence(self, q: 'Distribution') -> Union[nn.Variable, np.ndarray]:
        '''
        Compute the kullback leibler divergence between given distribution.
        This function will compute KL(self||q)

        Args:
            q(nnabla_rl.distributions.Distribution): target distribution to compute the kl_divergence

        Returns:
            Union[nn.Variable, np.ndarray]: Kullback leibler divergence

        Raises:
            ValueError: target distribution's type does not match with current distribution type.

        '''
        raise NotImplementedError
