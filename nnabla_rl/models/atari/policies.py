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

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.atari.shared_functions import PPOSharedFunctionHead
from nnabla_rl.models.policy import StochasticPolicy


class PPOPolicy(StochasticPolicy):
    '''
    Shared parameter function proposed used in PPO paper for atari environment.
    This network outputs the policy distribution.
    See: https://arxiv.org/pdf/1707.06347.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _head: PPOSharedFunctionHead
    _action_dim: int

    def __init__(self, head: PPOSharedFunctionHead, scope_name: str, action_dim: int):
        super(PPOPolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim
        self._head = head

    def pi(self, s: nn.Variable) -> Distribution:
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_pi"):
                z = NPF.affine(h, n_outmaps=self._action_dim,
                               w_init=RI.NormcInitializer(std=0.01))
        return D.Softmax(z=z)

    def _hidden(self, s: nn.Variable) -> nn.Variable:
        return self._head(s)


class ICML2015TRPOPolicy(StochasticPolicy):
    '''
    Policy network proposed in TRPO original paper for atari environment.
    This network outputs the value and policy distribution.
    See: https://arxiv.org/pdf/1502.05477.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(ICML2015TRPOPolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NF.tanh(NPF.convolution(
                    s, 16, (4, 4), stride=(2, 2)))
            with nn.parameter_scope("conv2"):
                h = NF.tanh(NPF.convolution(
                    h, 16, (4, 4), pad=(1, 1), stride=(2, 2)))
            h = NF.reshape(h, (batch_size, -1), inplace=False)
            with nn.parameter_scope("affine1"):
                h = NF.tanh(NPF.affine(h, 20))
            with nn.parameter_scope("affine2"):
                z = NPF.affine(h, self._action_dim)

        return D.Softmax(z=z)


class A3CPolicy(StochasticPolicy):
    '''
    Shared parameter function used in A3C paper for atari environment.
    See: https://arxiv.org/pdf/1602.01783.pdf
    '''

    def __init__(self, head, scope_name, state_shape, action_dim):
        super(A3CPolicy, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._action_dim = action_dim
        self._head = head

    def pi(self, s: nn.Variable) -> Distribution:
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_pi"):
                z = NPF.affine(h, n_outmaps=self._action_dim)
        return D.Softmax(z=z)

    def _hidden(self, s):
        assert s.shape[1:] == self._state_shape
        return self._head(s)
