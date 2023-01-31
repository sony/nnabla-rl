# Copyright 2023 Sony Group Corporation.
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

import nnabla as nn
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.model import Model


class DecisionTransformer(Model, metaclass=ABCMeta):
    """Base Decision Transformer class."""

    def __init__(self, scope_name: str, num_heads: int, embedding_dim: int):
        super().__init__(scope_name)
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim


class DeterministicDecisionTransformer(DecisionTransformer):
    @abstractmethod
    def pi(self, s: nn.Variable, a: nn.Variable, rtg: nn.Variable, t: nn.Variable) -> nn.Variable:
        """Compute action for given state, action, and return to go (rtg)

        Args:
            s (nn.Variable): state variable
            a (nn.Variable): action variable
            rtg (nn.Variable): return to go variable
            t (nn.Variable): timesteps variable


        Returns:
            nn.Variable: action for given state, action, and return to go (rtg)
        """
        raise NotImplementedError


class StochasticDecisionTransformer(DecisionTransformer):
    @abstractmethod
    def pi(self, s: nn.Variable, a: nn.Variable,  rtg: nn.Variable, t: nn.Variable) -> Distribution:
        """Compute action distribution for given state, action, and return to
        go (rtg)

        Args:
            s (nn.Variable): state variable
            a (nn.Variable): action variable
            rtg (nn.Variable): return to go variable
            t (nn.Variable): timesteps variable

        Returns:
            nn.Variable: action distribution for given state, action and return to go (rtg)
        """
        raise NotImplementedError
