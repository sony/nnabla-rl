# Copyright 2023,2024 Sony Group Corporation.
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
from typing import Any, Dict, Tuple

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.functions as RF
import nnabla_rl.initializers as RI
from nnabla_rl.distributions import Distribution, Gaussian
from nnabla_rl.models import VariationalAutoEncoder


class HyARVAE(VariationalAutoEncoder):
    """Variational Auto Encoder model proposed by Li et al.

    in the HyAR paper.
    See: https://arxiv.org/abs/2109.05490
    """

    def __init__(self, scope_name: str, state_dim, action_dim, encode_dim, embed_dim):
        super().__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._encode_dim = encode_dim

        self._class_num, self._latent_dim = action_dim
        self._embed_dim = embed_dim

    def __deepcopy__(self, memodict: Dict[Any, Any] = {}):
        # nn.Variable cannot be deepcopied directly
        return self.__class__(self._scope_name, self._state_dim, self._action_dim, self._encode_dim, self._embed_dim)

    def encode(self, x: nn.Variable, **kwargs) -> nn.Variable:
        latent_distribution = self.latent_distribution(x, **kwargs)
        return latent_distribution.sample()

    def encode_and_decode(self, x: nn.Variable, **kwargs) -> Tuple[Distribution, Any]:
        if "action" in kwargs:
            (d_action, _) = kwargs["action"]
            e = self.encode_discrete_action(d_action)
        elif "e" in kwargs:
            e = kwargs["e"]
        else:
            raise NotImplementedError

        latent_distribution = self.latent_distribution(x, e=e, state=kwargs["state"])
        z = latent_distribution.sample()
        reconstructed = self.decode(z, e=e, state=kwargs["state"])
        return latent_distribution, reconstructed

    def decode(self, z: Any, **kwargs) -> nn.Variable:
        state = kwargs["state"]
        if "action" in kwargs:
            (d_action, _) = kwargs["action"]
            action = self.encode_discrete_action(d_action)
        elif "e" in kwargs:
            action = kwargs["e"]
        else:
            raise NotImplementedError

        with nn.parameter_scope(self._scope_name):
            with nn.parameter_scope("decoder"):
                c = NF.concatenate(state, action)
                linear1_init = RI.HeUniform(inmaps=c.shape[1], outmaps=256, factor=1 / 3)
                c = NF.relu(NPF.affine(c, n_outmaps=256, name="linear1", w_init=linear1_init, b_init=linear1_init))
                linear2_init = RI.HeUniform(inmaps=z.shape[1], outmaps=256, factor=1 / 3)
                z = NF.relu(NPF.affine(z, n_outmaps=256, name="linear2", w_init=linear2_init, b_init=linear2_init))

                h = z * c
                linear3_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
                h = NF.relu(NPF.affine(h, n_outmaps=256, name="linear3", w_init=linear3_init, b_init=linear3_init))
                linear4_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
                h = NF.relu(NPF.affine(h, n_outmaps=256, name="linear4", w_init=linear4_init, b_init=linear4_init))

                linear5_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
                ds = NF.relu(NPF.affine(h, n_outmaps=256, name="linear5", w_init=linear5_init, b_init=linear5_init))
                linear6_init = RI.HeUniform(inmaps=ds.shape[1], outmaps=self._state_dim, factor=1 / 3)
                ds = NPF.affine(ds, n_outmaps=self._state_dim, name="linear6", w_init=linear6_init, b_init=linear6_init)

                linear7_init = RI.HeUniform(inmaps=h.shape[1], outmaps=self._latent_dim, factor=1 / 3)
                x = NPF.affine(h, n_outmaps=self._latent_dim, name="linear7", w_init=linear7_init, b_init=linear7_init)
        return NF.tanh(x), NF.tanh(ds)

    def decode_multiple(self, z, decode_num: int, **kwargs):
        raise NotImplementedError

    def latent_distribution(self, x: nn.Variable, **kwargs) -> Distribution:
        state = kwargs["state"]
        if "action" in kwargs:
            (d_action, _) = kwargs["action"]
            action = self.encode_discrete_action(d_action)
        elif "e" in kwargs:
            action = kwargs["e"]
        else:
            raise NotImplementedError

        with nn.parameter_scope(self._scope_name):
            with nn.parameter_scope("encoder"):
                c = NF.concatenate(state, action)
                linear1_init = RI.HeUniform(inmaps=c.shape[1], outmaps=256, factor=1 / 3)
                c = NF.relu(NPF.affine(c, n_outmaps=256, name="linear1", w_init=linear1_init, b_init=linear1_init))
                linear2_init = RI.HeUniform(inmaps=x.shape[1], outmaps=256, factor=1 / 3)
                x = NF.relu(NPF.affine(x, n_outmaps=256, name="linear2", w_init=linear2_init, b_init=linear2_init))

                h = x * c
                linear3_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
                h = NF.relu(NPF.affine(h, n_outmaps=256, name="linear3", w_init=linear3_init, b_init=linear3_init))
                linear4_init = RI.HeUniform(inmaps=h.shape[1], outmaps=256, factor=1 / 3)
                h = NF.relu(NPF.affine(h, n_outmaps=256, name="linear4", w_init=linear4_init, b_init=linear4_init))
                linear5_init = RI.HeUniform(inmaps=h.shape[1], outmaps=self._encode_dim * 2, factor=1 / 3)
                h = NPF.affine(
                    h, n_outmaps=self._encode_dim * 2, name="linear5", w_init=linear5_init, b_init=linear5_init
                )
                reshaped = NF.reshape(h, shape=(-1, 2, self._encode_dim))
                mean, ln_var = NF.split(reshaped, axis=1)
                ln_var = NF.clip_by_value(ln_var, min=-8, max=30)

        return Gaussian(mean, ln_var)

    def encode_discrete_action(self, action):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("embed"):
                embedding = NPF.embed(
                    action, n_inputs=self._class_num, n_features=self._embed_dim, initializer=NI.UniformInitializer()
                )
                embedding = NF.reshape(embedding, shape=(-1, self._embed_dim))
                embedding = NF.tanh(embedding)
        return embedding

    def decode_discrete_action(self, action_embedding):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("embed"):
                label_embedding = NPF.embed(
                    self._labels,
                    n_inputs=self._class_num,
                    n_features=self._embed_dim,
                    initializer=NI.UniformInitializer(),
                )
                label_embedding = NF.reshape(label_embedding, shape=(-1, self._class_num, self._embed_dim))
                label_embedding = NF.tanh(label_embedding)

                action_embedding = NF.reshape(action_embedding, shape=(action_embedding.shape[0], 1, self._embed_dim))
                similarity = -NF.sum(NF.squared_error(label_embedding, action_embedding), axis=-1)
                d_action = RF.argmax(similarity, axis=1, keepdims=True)
        return d_action

    @property
    def _labels(self) -> nn.Variable:
        labels = np.array([label for label in range(self._class_num)], dtype=np.int32)
        labels = np.reshape(labels, newshape=(1, self._class_num))
        return nn.Variable.from_numpy_array(labels)
