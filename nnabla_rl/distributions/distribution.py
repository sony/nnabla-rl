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

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import nnabla as nn


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, noise_clip: Optional[Tuple[float, float]] = None) -> nn.Variable:
        '''
        Sample a value from the distribution.
        If noise_clip is specified, the sampled value will be clipped in the given range.
        Applicability of noise_clip depends on underlying implementation.

        Args:
            noise_clip(Tuple[float, float], optional):
                float tuple of size 2 which contains the min and max value of the noise.

        Returns:
             nn.Variable: Sampled value
        '''
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        '''
        The number of dimensions of the distribution
        '''
        raise NotImplementedError

    def sample_multiple(self, num_samples: int, noise_clip: Optional[Tuple[float, float]] = None) -> nn.Variable:
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
             nn.Variable: Sampled value.
        '''
        raise NotImplementedError

    def choose_probable(self) -> nn.Variable:
        '''
        Compute the most probable action of the distribution

        Returns:
             nnabla.Variable: Probable action of the distribution
        '''
        raise NotImplementedError

    def mean(self) -> nn.Variable:
        '''
        Compute the mean of the distribution (if exist)

        Returns:
             nn.Variable: mean of the distribution

        Raises:
             NotImplementedError: The distribution does not have mean
        '''
        raise NotImplementedError

    def log_prob(self, x: nn.Variable) -> nn.Variable:
        '''
        Compute the log probability of given input

        Args:
            x (nn.Variable): Target value to compute the log probability

        Returns:
            nn.Variable: Log probability of given input
        '''
        raise NotImplementedError

    def sample_and_compute_log_prob(self, noise_clip: Optional[Tuple[float, float]] = None) \
            -> Tuple[nn.Variable, nn.Variable]:
        '''
        Sample a value from the distribution and compute its log probability.

        Args:
            noise_clip(Tuple[float, float], optional):
                float tuple of size 2 which contains the min and max value of the noise.

        Returns:
            Tuple[nn.Variable, nn.Variable]: Sampled value and its log probabilty
        '''
        raise NotImplementedError

    def entropy(self) -> nn.Variable:
        '''
        Compute the entropy of the distribution

        Returns:
            nn.Variable: Entropy of the distribution
        '''
        raise NotImplementedError

    def kl_divergence(self, q: 'Distribution') -> nn.Variable:
        '''
        Compute the kullback leibler divergence between given distribution.
        This function will compute KL(self||q)

        Args:
            q(nnabla_rl.distributions.Distribution): target distribution to compute the kl_divergence

        Returns:
            nn.Variable: Kullback leibler divergence

        Raises:
            ValueError: target distribution's type does not match with current distribution type.

        '''
        raise NotImplementedError
