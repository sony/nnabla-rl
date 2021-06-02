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
import nnabla.initializer as NI
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla.parameter import get_parameter_or_create
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models.policy import DeterministicPolicy, StochasticPolicy


class TD3Policy(DeterministicPolicy):
    '''
    Actor model proposed by S. Fujimoto in TD3 paper for mujoco environment.
    See: https://arxiv.org/abs/1802.09477
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _max_action_value: float

    def __init__(self, scope_name: str, action_dim: int, max_action_value: float):
        super(TD3Policy, self).__init__(scope_name)
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            linear1_init = RI.HeUniform(
                inmaps=s.shape[1], outmaps=400, factor=1/3)
            h = NPF.affine(s, n_outmaps=400, name="linear1",
                           w_init=linear1_init, b_init=linear1_init)
            h = NF.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = NPF.affine(h, n_outmaps=300, name="linear2",
                           w_init=linear2_init, b_init=linear2_init)
            h = NF.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=self._action_dim, factor=1/3)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear3",
                           w_init=linear3_init, b_init=linear3_init)
        return NF.tanh(h) * self._max_action_value


class SACPolicy(StochasticPolicy):
    '''
    Actor model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _clip_log_sigma: bool
    _min_log_sigma: float
    _max_log_sigma: float

    def __init__(self,
                 scope_name: str,
                 action_dim: int,
                 clip_log_sigma: bool = True,
                 min_log_sigma: float = -20.0,
                 max_log_sigma: float = 2.0):
        super(SACPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        self._clip_log_sigma = clip_log_sigma
        self._min_log_sigma = min_log_sigma
        self._max_log_sigma = max_log_sigma

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim*2, name="linear3")
            reshaped = NF.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_sigma = NF.split(reshaped, axis=1)
            assert mean.shape == ln_sigma.shape
            assert mean.shape == (s.shape[0], self._action_dim)
            if self._clip_log_sigma:
                ln_sigma = NF.clip_by_value(
                    ln_sigma, min=self._min_log_sigma, max=self._max_log_sigma)
            ln_var = ln_sigma * 2.0
        return D.SquashedGaussian(mean=mean, ln_var=ln_var)


class BEARPolicy(StochasticPolicy):
    '''
    Actor model proposed by A. Kumar, et al. in BEAR paper for mujoco environment.
    See: https://arxiv.org/pdf/1906.00949.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(BEARPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            linear1_init = RI.HeUniform(
                inmaps=s.shape[1], outmaps=400, factor=1/3)
            h = NPF.affine(s, n_outmaps=400, name="linear1",
                           w_init=linear1_init, b_init=linear1_init)
            h = NF.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = NPF.affine(h, n_outmaps=300, name="linear2",
                           w_init=linear2_init, b_init=linear2_init)
            h = NF.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=self._action_dim*2, factor=1/3)
            h = NPF.affine(h, n_outmaps=self._action_dim*2, name="linear3",
                           w_init=linear3_init, b_init=linear3_init)
            reshaped = NF.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_var = NF.split(reshaped, axis=1)
            assert mean.shape == ln_var.shape
            assert mean.shape == (s.shape[0], self._action_dim)
        return D.Gaussian(mean=mean, ln_var=ln_var)


class PPOPolicy(StochasticPolicy):
    '''
    Actor model proposed by John Schulman, et al. in PPO paper for mujoco environment.
    This network outputs the policy distribution
    See: https://arxiv.org/pdf/1707.06347.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(PPOPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=64, name="linear1",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=64, name="linear2",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            mean = NPF.affine(h, n_outmaps=self._action_dim, name="linear3",
                              w_init=RI.NormcInitializer(std=0.01))
            ln_sigma = nn.parameter.get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=NI.ConstantInitializer(0.))
            ln_var = NF.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
            assert mean.shape == ln_var.shape
            assert mean.shape == (s.shape[0], self._action_dim)
        return D.Gaussian(mean=mean, ln_var=ln_var)


class ICML2015TRPOPolicy(StochasticPolicy):
    '''
    Actor model proposed by John Schulman, et al. in TRPO paper for mujoco environment.
    See: https://arxiv.org/pdf/1502.05477.pdf (Original paper)
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(ICML2015TRPOPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=30, name="linear1")
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim*2, name="linear2")
            reshaped = NF.reshape(
                h, shape=(-1, 2, self._action_dim), inplace=False)
            mean, ln_sigma = NF.split(reshaped, axis=1)
            assert mean.shape == ln_sigma.shape
            assert mean.shape == (s.shape[0], self._action_dim)
            ln_var = ln_sigma * 2.0
        return D.Gaussian(mean=mean, ln_var=ln_var)


class TRPOPolicy(StochasticPolicy):
    '''
    Actor model proposed by Peter Henderson, et al.
    in Deep Reinforcement Learning that Matters paper for mujoco environment.
    See: https://arxiv.org/abs/1709.06560.pdf
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(TRPOPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=64, name="linear1",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=64, name="linear2",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            mean = NPF.affine(h, n_outmaps=self._action_dim, name="linear3",
                              w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            assert mean.shape == (s.shape[0], self._action_dim)

            ln_sigma = get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=NI.ConstantInitializer(0.))
            ln_var = NF.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
        return D.Gaussian(mean, ln_var)


class GAILPolicy(StochasticPolicy):
    '''
    Actor model proposed by Jonathan Ho, et al.
    See: https://arxiv.org/pdf/1606.03476.pdf
    '''

    def __init__(self, scope_name: str, action_dim: str):
        super(GAILPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=100, name="linear1",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=100, name="linear2",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            mean = NPF.affine(h, n_outmaps=self._action_dim, name="linear3",
                              w_init=RI.NormcInitializer(std=0.01))
            assert mean.shape == (s.shape[0], self._action_dim)

            ln_sigma = get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=NI.ConstantInitializer(0.))
            ln_var = NF.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
        return D.Gaussian(mean, ln_var)
