# Copyright 2024 Sony Group Corporation.
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


class IntraPolicy(Model, metaclass=ABCMeta):
    def __init__(self, scope_name: str):
        super(IntraPolicy, self).__init__(scope_name)


class StochasticIntraPolicy(IntraPolicy, metaclass=ABCMeta):
    """StochasticIntraPolicy Abstract class for stochastic intra policy.

    This policy returns a probability distribution of action for the
    given state and option.
    """

    @abstractmethod
    def intra_pi(self, state: nn.Variable, option: nn.Variable) -> Distribution:
        """intra_pi.

        Args:
            state (nnabla.Variable): State variable
            option (nnabla.Variable): Option variable

        Returns:
            nnabla_rl.distributions.Distribution: Probability distribution of the action for the given state and option
        """
        raise NotImplementedError
