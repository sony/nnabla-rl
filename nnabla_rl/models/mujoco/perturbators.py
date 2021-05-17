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
from nnabla_rl.models.perturbator import Perturbator


class BCQPerturbator(Perturbator):
    '''
    Perturbator model proposed by S. Fujimoto in BCQ paper for mujoco environment.
    See: https://arxiv.org/abs/1812.02900
    '''

    def __init__(self, scope_name, state_dim, action_dim, max_action_value):
        super(BCQPerturbator, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def generate_noise(self, s, a, phi):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            h = NPF.affine(h, n_outmaps=400, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=300, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear3")
        return NF.tanh(h) * self._max_action_value * phi
