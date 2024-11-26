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

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla_rl.distributions as D
from nnabla.parameter import get_parameter_or_create
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.atari.shared_functions import OptionCriticSharedFunctionHead
from nnabla_rl.models.intra_policy import StochasticIntraPolicy


class AtariOptionCriticIntraPolicy(StochasticIntraPolicy):

    def __init__(
        self,
        head: OptionCriticSharedFunctionHead,
        scope_name: str,
        num_options: int,
        action_dim: int,
        temperature: float = 1.0,
    ):
        super(AtariOptionCriticIntraPolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim
        self._num_options = num_options
        self._head = head
        self._temperature = temperature

    def intra_pi(self, state: nn.Variable, option: nn.Variable) -> Distribution:
        h = self._hidden(state)
        h.need_grad = False

        assert len(h.shape) == 2
        batch_size = h.shape[0]
        h_dim = h.shape[1]
        assert batch_size == option.shape[0]
        mask = NF.one_hot(NF.reshape(option, (-1, 1), inplace=False), (self._num_options,))

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_intra_pi"):
                w = get_parameter_or_create(
                    "w",
                    (1, self._num_options, h_dim, self._action_dim),
                    initializer=NI.UniformInitializer((-1.0, 1.0)),
                )
                b = get_parameter_or_create(
                    "b", (1, self._num_options, self._action_dim), initializer=NI.ConstantInitializer(0.0)
                )

                # multiply w and b by the mask and then sum over the options dimension (axis=1)
                option_w = NF.sum(w * NF.reshape(mask, (batch_size, self._num_options, 1, 1)), axis=1)
                option_b = NF.sum(b * NF.reshape(mask, (batch_size, self._num_options, 1)), axis=1)

                # z.shape = (batch_size, action_dim)
                z = (
                    NF.reshape(
                        NF.batch_matmul(NF.reshape(h, (batch_size, 1, h_dim)), option_w), (batch_size, self._action_dim)
                    )
                    + option_b
                )

        return D.Softmax(z=z / self._temperature)

    def _hidden(self, state):
        return self._head(state)
