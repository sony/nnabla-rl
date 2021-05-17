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

from typing import Tuple

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.functions as RF
from nnabla_rl.distributions import Distribution
from nnabla_rl.models.encoder import VariationalAutoEncoder


class UnsquashedVariationalAutoEncoder(VariationalAutoEncoder):
    '''
    Almost identical to BCQ style variational auto encoder proposed by S. Fujimoto in BCQ paper for mujoco environment.
    See: https://arxiv.org/pdf/1812.02900.pdf
    The main difference is that the output action is not squashed with tanh for computational convenience.
    '''

    def __init__(self, scope_name, state_dim, action_dim, latent_dim):
        super(UnsquashedVariationalAutoEncoder, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._latent_dim = latent_dim

    def encode_and_decode(self, x: nn.Variable, **kwargs) -> Tuple[Distribution, nn.Variable]:
        '''
        Args:
            x (nn.Variable): encoder input.

        Returns:
            [Distribution, nn.Variable]: Reconstructed input and latent distribution
        '''
        a = kwargs['action']
        h = NF.concatenate(x, a)
        latent_distribution = self.latent_distribution(h)
        z = latent_distribution.sample()
        reconstructed = self.decode(z, state=x)
        return latent_distribution, reconstructed

    def encode(self, x: nn.Variable, **kwargs) -> nn.Variable:
        a = kwargs['action']
        x = NF.concatenate(x, a)
        latent_distribution = self.latent_distribution(x)
        return latent_distribution.sample()

    def decode(self, z: nn.Variable, **kwargs) -> nn.Variable:
        s = kwargs['state']
        if z is None:
            z = NF.randn(shape=(s.shape[0], self._latent_dim))
            z = NF.clip_by_value(z, -0.5, 0.5)
        with nn.parameter_scope(self.scope_name):
            x = NF.concatenate(s, z)
            h = NPF.affine(x, n_outmaps=750, name="linear4")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear5")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear6")
        return h

    def decode_multiple(self, z: nn.Variable, decode_num: int, **kwargs) -> nn.Variable:
        s = kwargs['state']
        if z is None:
            z = NF.randn(shape=(s.shape[0], decode_num, self._latent_dim))
            z = NF.clip_by_value(z, -0.5, 0.5)
        s = RF.expand_dims(s, axis=0)
        s = RF.repeat(s, repeats=decode_num, axis=0)
        s = NF.transpose(s, axes=(1, 0, 2))
        assert s.shape[:-1] == z.shape[:-1]

        x = NF.concatenate(s, z, axis=2)
        x = NF.reshape(x, shape=(-1, x.shape[-1]))
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(x, n_outmaps=750, name="linear4")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear5")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear6")
            h = NF.reshape(h, shape=(-1, decode_num, h.shape[-1]))
        return h

    def latent_distribution(self, x: nn.Variable, **kwargs) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(x, n_outmaps=750, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._latent_dim*2, name="linear3")
            reshaped = NF.reshape(h, shape=(-1, 2, self._latent_dim))
            mean, ln_var = NF.split(reshaped, axis=1)
            # Clip for numerical stability
            ln_var = NF.clip_by_value(ln_var, min=-8, max=30)
        return D.Gaussian(mean, ln_var)


class BCQVariationalAutoEncoder(UnsquashedVariationalAutoEncoder):
    '''
    BCQ style variational auto encoder proposed by S. Fujimoto in BCQ paper for mujoco environment.
    See: https://arxiv.org/pdf/1812.02900.pdf
    '''

    def __init__(self, scope_name, state_dim, action_dim, latent_dim, max_action_value):
        super(BCQVariationalAutoEncoder, self).__init__(scope_name, state_dim, action_dim, latent_dim)
        self._max_action_value = max_action_value

    def decode(self, z: nn.Variable, **kwargs) -> nn.Variable:
        unsquashed = super(BCQVariationalAutoEncoder, self).decode(z, **kwargs)
        return NF.tanh(unsquashed) * self._max_action_value

    def decode_multiple(self, z: nn.Variable, decode_num: int, **kwargs) -> nn.Variable:
        unsquashed = super(BCQVariationalAutoEncoder, self).decode_multiple(z, decode_num, **kwargs)
        return NF.tanh(unsquashed) * self._max_action_value
