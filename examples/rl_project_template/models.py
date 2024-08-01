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
import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.distributions as D
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.models import ContinuousQFunction, StochasticPolicy


class TemplateQFunction(ContinuousQFunction):
    # This is a sample QFunction model to be used in Soft-Actor Critic(SAC) algorithm.
    # Model to implement depends on the RL algorithm.
    def __init__(self, scope_name: str):
        super().__init__(scope_name=scope_name)

    def q(self, state: nn.Variable, action: nn.Variable) -> nn.Variable:
        # Using the input state and action, here we implemented a simple model that outputs
        # q-value (1 dim) using these inputs.
        # Modify this model to better perform on your environment.
        # The state's and action's shape are both (batch size, variable's shape)
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(state, action)
            h = NPF.affine(h, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3")
        return h


class TemplatePolicy(StochasticPolicy):
    # This is a sample stochstic policy model to be used in Soft-Actor Critic(SAC) algorithm.
    # Stochastic policy is a type of policy that outputs the action's probability distribution
    # instead of action itself.
    # Model to implement depends on the RL algorithm.
    def __init__(self, scope_name: str, action_dim=1):
        super().__init__(scope_name=scope_name)
        self._action_dim = action_dim

    def pi(self, state: nn.Variable) -> Distribution:
        # The state's shape is (batch size, state's shape)
        # state's shape is same as the one set in the environment's implementation.

        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(state, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim * 2, name="linear3")
            reshaped = NF.reshape(h, shape=(-1, 2, self._action_dim))

            # Split the output into mean and variance of the Gaussian distribution.
            mean, ln_var = NF.split(reshaped, axis=1)

            # Check that output shape is as expected
            assert mean.shape == ln_var.shape
            assert mean.shape == (state.shape[0], self._action_dim)
        # SquashedGaussian is a distribution that applies tanh to the output of Gaussian distribution.
        return D.SquashedGaussian(mean=mean, ln_var=ln_var)
