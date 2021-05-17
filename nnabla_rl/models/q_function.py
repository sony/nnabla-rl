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

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.models.model import Model


class QFunction(Model, metaclass=ABCMeta):
    """Base QFunction Class
    """
    @abstractmethod
    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        """Compute Q-value for given state and action

        Args:
            s (nn.Variable): state variable
            a (nn.Variable): action variable

        Returns:
            nn.Variable: Q-value for given state and action
        """
        raise NotImplementedError

    def all_q(self, s: nn.Variable) -> nn.Variable:
        """Compute Q-values for each action for given state

        Args:
            s (nn.Variable): state variable

        Returns:
            nn.Variable: Q-values for each action for given state
        """
        raise NotImplementedError

    def max_q(self, s: nn.Variable) -> nn.Variable:
        """Compute maximum Q-value for given state

        Args:
            s (nn.Variable): state variable

        Returns:
            nn.Variable: maximum Q-value value for given state
        """
        raise NotImplementedError

    def argmax_q(self, s: nn.Variable) -> nn.Variable:
        """Compute the action which maximizes the Q-value for given state

        Args:
            s (nn.Variable): state variable

        Returns:
            nn.Variable: action which maximizes the Q-value for given state
        """
        raise NotImplementedError


class DiscreteQFunction(QFunction):
    """Base QFunction Class for discrete action environment
    """
    @abstractmethod
    def all_q(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        q_values = self.all_q(s)
        q_value = NF.sum(q_values * NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],)),
                         axis=1,
                         keepdims=True)  # get q value of a

        return q_value

    def max_q(self, s: nn.Variable) -> nn.Variable:
        q_values = self.all_q(s)
        return NF.max(q_values, axis=1, keepdims=True)

    def argmax_q(self, s: nn.Variable) -> nn.Variable:
        q_values = self.all_q(s)
        return RF.argmax(q_values, axis=1, keepdims=True)


class ContinuousQFunction(QFunction):
    """Base QFunction Class for continuous action environment
    """
    pass
