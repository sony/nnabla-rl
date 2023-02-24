# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from typing import Optional, Tuple

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import DeterministicPolicy
from nnabla_rl.models.q_function import ContinuousQFunction, FactoredContinuousQFunction


class TD3QFunction(ContinuousQFunction):
    """Critic model proposed by S.

    Fujimoto in TD3 paper for mujoco environment.
    See: https://arxiv.org/abs/1802.09477
    """

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
    """QFunciton model proposed by T.

    Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    """

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


class SACDQFunction(FactoredContinuousQFunction):
    """Factored QFunciton model proposed by J.

    MacGlashan in SAC-D paper for mujoco environment.
    See: https://arxiv.org/abs/2206.13901
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _num_factors: int
    _optimal_policy: Optional[DeterministicPolicy]

    def __init__(self, scope_name, num_factors: int, optimal_policy: Optional[DeterministicPolicy] = None):
        super(SACDQFunction, self).__init__(scope_name)
        self._num_factors = num_factors
        self._optimal_policy = optimal_policy

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        return NF.sum(self.factored_q(s, a), axis=1, keepdims=True)

    def factored_q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            h = NPF.affine(h, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._num_factors, name="linear3")
        return h

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)

    @property
    def num_factors(self) -> int:
        return self._num_factors


class HERQFunction(ContinuousQFunction):
    def __init__(self, scope_name: str, optimal_policy: Optional[DeterministicPolicy] = None):
        super(HERQFunction, self).__init__(scope_name)
        self._optimal_policy = optimal_policy

    def q(self, s: Tuple[nn.Variable, nn.Variable, nn.Variable], a: nn.Variable) -> nn.Variable:
        obs, goal, _ = s
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(obs, goal, a, axis=1)
            linear1_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = NPF.affine(h, n_outmaps=64, name='linear1', w_init=linear1_init)
            h = NF.relu(h)
            linear2_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = NPF.affine(h, n_outmaps=64, name='linear2', w_init=linear2_init)
            h = NF.relu(h)
            linear3_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = NPF.affine(h, n_outmaps=64, name='linear3', w_init=linear3_init)
            h = NF.relu(h)
            pred_q_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=1)
            q = NPF.affine(h, n_outmaps=1, name='pred_q', w_init=pred_q_init)
        return q

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)


class XQLQFunction(ContinuousQFunction):
    """QFunction model used in the training of XQL.

    Used by D. Garg et al. for experiments in mujoco environment.
    Same as the SACQFunction except for the initializer.
    See: https://github.com/Div99/XQL
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _optimal_policy: Optional[DeterministicPolicy]

    def __init__(self, scope_name, optimal_policy: Optional[DeterministicPolicy] = None):
        super(XQLQFunction, self).__init__(scope_name)
        self._optimal_policy = optimal_policy

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        w_init = NI.OrthogonalInitializer(np.sqrt(2.0))
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            h = NPF.affine(h, n_outmaps=256, name="linear1", w_init=w_init)
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2", w_init=w_init)
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3", w_init=w_init)
        return h

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)
