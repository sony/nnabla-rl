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

from typing import Optional

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import DeterministicPolicy
from nnabla_rl.models.q_function import ContinuousQFunction


class TD3QFunction(ContinuousQFunction):
    '''
    Critic model proposed by S. Fujimoto in TD3 paper for mujoco environment.
    See: https://arxiv.org/abs/1802.09477
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _optimal_policy: Optional[DeterministicPolicy]

    def __init__(self, scope_name: str, optimal_policy: Optional[DeterministicPolicy] = None):
        super(TD3QFunction, self).__init__(scope_name)
        self._optimal_policy = optimal_policy

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            linear1_init = RI.HeUniform(
                inmaps=h.shape[1], outmaps=400, factor=1/3)
            h = NPF.affine(h, n_outmaps=400, name="linear1",
                           w_init=linear1_init, b_init=linear1_init)
            h = NF.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = NPF.affine(h, n_outmaps=300, name="linear2",
                           w_init=linear2_init, b_init=linear2_init)
            h = NF.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=1, factor=1/3)
            h = NPF.affine(h, n_outmaps=1, name="linear3",
                           w_init=linear3_init, b_init=linear3_init)
        return h

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)


class SACQFunction(ContinuousQFunction):
    '''
    QFunciton model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _optimal_policy: Optional[DeterministicPolicy]

    def __init__(self, scope_name, optimal_policy: Optional[DeterministicPolicy] = None):
        super(SACQFunction, self).__init__(scope_name)
        self._optimal_policy = optimal_policy

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            h = NPF.affine(h, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3")
        return h

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)
