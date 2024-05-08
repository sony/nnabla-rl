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

from typing import Tuple

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.reward_function import RewardFunction


class AMPDiscriminator(RewardFunction):
    """Discriminator model used as reward function proposed by Xue Bin Peng, et
    al.

    See: https://arxiv.org/abs/2104.02180
    """

    def __init__(self, scope_name: str, output_layer_initializer_scale: float):
        super(AMPDiscriminator, self).__init__(scope_name)
        assert output_layer_initializer_scale > 0.0, f"{output_layer_initializer_scale} should be larger than 0.0"
        self._output_layer_initializer_scale = output_layer_initializer_scale

    def r(self, s_current: Tuple[nn.Variable, ...], a_current: nn.Variable, s_next: Tuple[nn.Variable, ...]
          ) -> nn.Variable:
        assert len(s_current) == 7 or len(s_current) == 3
        for _s in s_current[1:]:
            assert s_current[0].shape[0] == _s.shape[0]

        s_for_reward = s_current[1]

        # NOTE: s_for_reward has s and s_next.
        # See author's enviroment implmentation and our env wrapper implementation.
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s_for_reward,
                           n_outmaps=1024,
                           name="linear1",
                           w_init=RI.GlorotUniform(inmaps=s_for_reward.shape[1], outmaps=1024))
            h = NF.relu(x=h)
            h = NPF.affine(h,
                           n_outmaps=512,
                           name="linear2",
                           w_init=RI.GlorotUniform(inmaps=h.shape[1], outmaps=512))
            h = NF.relu(x=h)
            h = NPF.affine(h,
                           n_outmaps=1,
                           name="logits",
                           w_init=NI.UniformInitializer((-1.0 * self._output_layer_initializer_scale,
                                                         self._output_layer_initializer_scale)))
        return h
