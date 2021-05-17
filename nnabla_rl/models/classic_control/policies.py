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

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import StochasticPolicy


class REINFORCEDiscretePolicy(StochasticPolicy):
    '''
    REINFORCE policy for classic control discrete environment.
    This network outputs the policy distribution.
    See: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(REINFORCEDiscretePolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=200, name="linear1",
                           w_init=RI.HeNormal(s.shape[1], 200))
            h = NF.leaky_relu(h)
            h = NPF.affine(h, n_outmaps=200, name="linear2",
                           w_init=RI.HeNormal(s.shape[1], 200))
            h = NF.leaky_relu(h)
            z = NPF.affine(h, n_outmaps=self._action_dim,
                           name="linear3", w_init=RI.LeCunNormal(s.shape[1], 200))

        return D.Softmax(z=z)


class REINFORCEContinousPolicy(StochasticPolicy):
    '''
    REINFORCE policy for classic control continous environment.
    This network outputs the policy distribution.
    See: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _fixed_ln_var: np.ndarray

    def __init__(self, scope_name: str, action_dim: int, fixed_ln_var: np.ndarray):
        super(REINFORCEContinousPolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim
        if np.isscalar(fixed_ln_var):
            self._fixed_ln_var = np.full(self._action_dim, fixed_ln_var)
        else:
            self._fixed_ln_var = fixed_ln_var

    def pi(self, s: nn.Variable) -> nn.Variable:
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=200, name="linear1",
                           w_init=RI.HeNormal(s.shape[1], 200))
            h = NF.leaky_relu(h)
            h = NPF.affine(h, n_outmaps=200, name="linear2",
                           w_init=RI.HeNormal(s.shape[1], 200))
            h = NF.leaky_relu(h)
            z = NPF.affine(h, n_outmaps=self._action_dim,
                           name="linear3", w_init=RI.HeNormal(s.shape[1], 200))

        return D.Gaussian(z, np.tile(self._fixed_ln_var, (batch_size, 1)))
