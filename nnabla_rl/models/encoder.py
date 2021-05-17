# Copyright 2021 Sony Corporation.
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
from nnabla_rl.distributions import Distribution
from nnabla_rl.models.model import Model


class Encoder(Model, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, x: nn.Variable, **kwargs) -> nn.Variable:
        '''
        Encode the input variable to latent representation.

        Args:
            x (nn.Variable): encoder input.

        Returns:
            nn.Variable: latent variable
        '''
        raise NotImplementedError


class VariationalAutoEncoder(Encoder):
    @abstractmethod
    def encode_and_decode(self, x: nn.Variable, **kwargs) -> Tuple[Distribution, nn.Variable]:
        '''
        Encode the input variable and reconstruct.

        Args:
            x (nn.Variable): encoder input.

        Returns:
            Tuple[Distribution, nn.Variable]: latent distribution and reconstructed input
        '''
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: Optional[nn.Variable], **kwargs) -> nn.Variable:
        '''
        Reconstruct the latent representation.

        Args:
            z (nn.Variable, optional): latent variable. If the input is None, random sample will be used instead.

        Returns:
            nn.Variable: reconstructed variable
        '''
        raise NotImplementedError

    @abstractmethod
    def decode_multiple(self, z: Optional[nn.Variable], decode_num: int, **kwargs):
        '''
        Reconstruct multiple latent representations.

        Args:
            z (nn.Variable, optional): encoder input. If the input is None, random sample will be used instead.

        Returns:
            nn.Variable: Reconstructed input and latent distribution
        '''
        raise NotImplementedError

    @abstractmethod
    def latent_distribution(self, x: nn.Variable, **kwargs) -> Distribution:
        '''
        Compute the latent distribution :math:`p(z|x)`.

        Args:
            x (nn.Variable): encoder input.

        Returns:
            Distribution: latent distribution
        '''
        raise NotImplementedError
