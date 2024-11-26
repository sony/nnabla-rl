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
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.models.model import Model


class OptionValueFunction(Model, metaclass=ABCMeta):
    """Base OptionValueFunction Class."""

    @abstractmethod
    def option_v(self, state: nn.Variable, option: nn.Variable) -> nn.Variable:
        """Compute value for given state and option.

        Args:
            state (nn.Variable): state variable
            option (nn.Variable): option variable

        Returns:
            nn.Variable: value for given state and option
        """
        raise NotImplementedError

    def all_option_v(self, state: nn.Variable) -> nn.Variable:
        """Compute values for each option for given state.

        Args:
            state (nn.Variable): state variable

        Returns:
            nn.Variable: values for each option for given state
        """
        raise NotImplementedError

    def max_option_v(self, state: nn.Variable) -> nn.Variable:
        """Compute maximum value for given state.

        Args:
            state (nn.Variable): state variable

        Returns:
            nn.Variable: maximum value for given state
        """
        raise NotImplementedError

    def argmax_option_v(self, state: nn.Variable) -> nn.Variable:
        """Compute the option which maximizes the value for given state.

        Args:
            state (nn.Variable): state variable

        Returns:
            nn.Variable: option which maximizes the value for given state
        """
        raise NotImplementedError


class DiscreteOptionValueFunction(OptionValueFunction):
    """Discrete Option Value Function class."""

    @abstractmethod
    def all_option_v(self, state: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def option_v(self, state: nn.Variable, option: nn.Variable) -> nn.Variable:
        option_values = self.all_option_v(state)
        option_value = NF.sum(
            option_values * NF.one_hot(NF.reshape(option, (-1, 1), inplace=False), (option_values.shape[1],)),
            axis=1,
            keepdims=True,
        )  # get q value of option

        return option_value

    def max_option_v(self, state: nn.Variable) -> nn.Variable:
        option_values = self.all_option_v(state)
        return NF.max(option_values, axis=1, keepdims=True)

    def argmax_option_v(self, state: nn.Variable) -> nn.Variable:
        option_values = self.all_option_v(state)
        return RF.argmax(option_values, axis=1, keepdims=True)

    def expectation_option_v(self, state: nn.Variable, probs: nn.Variable) -> nn.Variable:
        option_values = self.all_option_v(state)
        assert len(option_values.shape) == 1
        assert len(option_values.shape[1]) == len(probs)

        option_values = option_values * NF.reshape(probs, (1, -1))
        return NF.sum(option_values, axis=1, keepdims=True)
