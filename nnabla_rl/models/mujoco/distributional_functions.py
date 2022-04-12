# Copyright 2022 Sony Group Corporation.
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
from nnabla_rl.models import ContinuousQuantileDistributionFunction


class QRSACQuantileDistributionFunction(ContinuousQuantileDistributionFunction):
    '''
    Example quantile distribution function model designed for nnabla_rl's evaluation in mujoco environment.
    '''

    def __init__(self, scope_name, n_quantile):
        super().__init__(scope_name, n_quantile)

    def quantiles(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            h = NPF.affine(h, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            quantiles = NPF.affine(h, n_outmaps=self._n_quantile, name="linear3")
        return quantiles
