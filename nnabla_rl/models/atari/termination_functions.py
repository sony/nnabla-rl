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
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.atari.shared_functions import OptionCriticSharedFunctionHead
from nnabla_rl.models.termination_function import StochasticTerminationFunction


class AtariOptionCriticTerminationFunction(StochasticTerminationFunction):
    _head: OptionCriticSharedFunctionHead
    _num_options: int

    def __init__(self, head: OptionCriticSharedFunctionHead, scope_name: str, num_options: int):
        super(AtariOptionCriticTerminationFunction, self).__init__(scope_name=scope_name)
        self._num_options = num_options
        self._head = head

    def termination(self, state: nn.Variable, option: nn.Variable) -> Distribution:
        h = self._hidden(state)
        h.need_grad = False
        option_mask = NF.one_hot(NF.reshape(option, (-1, 1), inplace=False), (self._num_options,))

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_termination"):
                z = NPF.affine(
                    h,
                    n_outmaps=self._num_options,
                    w_init=RI.GlorotUniform(h.shape[1], self._num_options),
                    b_init=NI.ConstantInitializer(0.1),
                )
                option_z = NF.sum(z * option_mask, axis=1, keepdims=True)

        return D.Bernoulli(z=option_z)  # return (batch_size, num_options)

    def _hidden(self, state: nn.Variable) -> nn.Variable:
        return self._head(state)
