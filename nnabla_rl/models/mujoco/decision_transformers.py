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

from typing import Optional

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl as rl
import nnabla_rl.functions as RF
import nnabla_rl.parametric_functions as RPF
from nnabla_rl.models import DeterministicDecisionTransformer
from nnabla_rl.utils.misc import create_attention_mask


class MujocoDecisionTransformer(DeterministicDecisionTransformer):
    def __init__(self,
                 scope_name: str,
                 action_dim: int,
                 max_timestep: int,
                 context_length: int,
                 num_heads: int = 1,
                 embedding_dim: int = 128):
        super().__init__(scope_name, num_heads, embedding_dim)
        self._action_dim = action_dim
        self._max_timestep = max_timestep
        self._context_length = context_length
        self._attention_layers = 3

    def pi(self, s: nn.Variable, a: nn.Variable, rtg: nn.Variable, t: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            rtg_embedding = self._embed_rtg(rtg)
            s_embedding = self._embed_state(s)
            if a is None or a.shape[1] != s.shape[1]:
                dummy = nn.Variable.from_numpy_array(np.empty((s.shape[0], 1, self._action_dim)))
                if a is None:
                    a_embedding = self._embed_action(dummy)
                else:
                    a_embedding = self._embed_action(NF.concatenate(a, dummy, axis=1))
            else:
                a_embedding = self._embed_action(a)

            token_embedding = RF.concat_interleave((rtg_embedding, s_embedding, a_embedding), axis=1)
            if a is None or a.shape[1] != s.shape[1]:
                token_embedding = token_embedding[:, :-1, :]

            position_embedding = self._embed_t(t, self._max_timestep + 1)
            x = token_embedding + position_embedding
            with nn.parameter_scope('layer_norm'):
                fix_parameters = rl.is_eval_scope()
                # 0.003 is almost sqrt(10^-5)
                x = NPF.layer_normalization(x, batch_axis=(0, 1), fix_parameters=fix_parameters, eps=0.003)

            block_size = token_embedding.shape[1]
            attention_mask = create_attention_mask(block_size, block_size)
            for i in range(self._attention_layers):
                with nn.parameter_scope(f'attention_block{i}'):
                    x = self._attention_block(x, attention_mask=attention_mask)

            # Use predictions from state embeddings
            x = x[:, 1::3, :]
            with nn.parameter_scope('affine'):
                actions = NPF.affine(x, n_outmaps=self._action_dim, base_axis=2)
        return NF.tanh(actions)

    def _embed_state(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope("state_embedding"):
            with nn.parameter_scope("linear"):
                return NPF.affine(s, n_outmaps=self._embedding_dim, base_axis=2)

    def _embed_action(self, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope("action_embedding"):
            with nn.parameter_scope("linear"):
                return NPF.affine(a, n_outmaps=self._embedding_dim, base_axis=2)

    def _embed_rtg(self, rtg: nn.Variable) -> nn.Variable:
        with nn.parameter_scope("rtg_embedding"):
            with nn.parameter_scope("linear"):
                return NPF.affine(rtg, n_outmaps=self._embedding_dim, base_axis=2)

    def _embed_t(self, timesteps: nn.Variable, T: int) -> nn.Variable:
        """T stands for max timestep among trajectories in the dataset."""
        with nn.parameter_scope("t_embedding"):
            embedding = NPF.embed(timesteps, n_inputs=T, n_features=self._embedding_dim)
            return NF.reshape(embedding, shape=(embedding.shape[0], embedding.shape[1], -1))

    def _attention_block(self, x: nn.Variable, attention_mask=None) -> nn.Variable:
        with nn.parameter_scope('layer_norm1'):
            fix_parameters = rl.is_eval_scope()
            normalized_x1 = NPF.layer_normalization(x, batch_axis=(0, 1), fix_parameters=fix_parameters, eps=0.003)
        with nn.parameter_scope('causal_self_attention'):
            attention_dropout = None if rl.is_eval_scope() else 0.1
            output_dropout = None if rl.is_eval_scope() else 0.1
            x = x + RPF.causal_self_attention(normalized_x1,
                                              embed_dim=self._embedding_dim,
                                              num_heads=self._num_heads,
                                              mask=attention_mask,
                                              attention_dropout=attention_dropout,
                                              output_dropout=output_dropout)
        with nn.parameter_scope('layer_norm2'):
            fix_parameters = rl.is_eval_scope()
            normalized_x2 = NPF.layer_normalization(x, batch_axis=(0, 1), fix_parameters=fix_parameters, eps=0.003)
        with nn.parameter_scope('mlp'):
            block_dropout = None if rl.is_eval_scope() else 0.1
            x = x + self._block_mlp(normalized_x2, block_dropout)
        return x

    def _block_mlp(self, x: nn.Variable, dropout: Optional[float] = None) -> nn.Variable:
        # NOTE: original code uses Conv1D operation defined below but it's same as affine layer.
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L91
        with nn.parameter_scope('linear1'):
            x = NPF.affine(x, n_outmaps=4 * self._embedding_dim, base_axis=2, w_init=NI.NormalInitializer(0.02))
            # Relu is used for mujoco
            x = NF.relu(x)
        with nn.parameter_scope('linear2'):
            x = NPF.affine(x, n_outmaps=self._embedding_dim, base_axis=2, w_init=NI.NormalInitializer(0.02))
        if dropout is not None:
            x = NF.dropout(x, p=dropout)
        return x
