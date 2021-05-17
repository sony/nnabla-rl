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
import pytest

import nnabla as nn
from nnabla_rl.models.mujoco.policies import SACPolicy, TD3Policy


class TestTD3Policy(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_scope_name(self):
        action_dim = 5
        scope_name = "test"
        model = TD3Policy(scope_name=scope_name,
                          action_dim=action_dim,
                          max_action_value=1.0)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = TD3Policy(scope_name=scope_name,
                          action_dim=action_dim,
                          max_action_value=1.0)
        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, state_dim))
        model.pi(input_state)

        assert len(model.get_parameters()) == 6


class TestSACPolicy(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_scope_name(self):
        action_dim = 5
        scope_name = "test"
        model = SACPolicy(scope_name=scope_name,
                          action_dim=action_dim)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = SACPolicy(scope_name=scope_name,
                          action_dim=action_dim)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, state_dim))
        model.pi(input_state)

        assert len(model.get_parameters()) == 6


if __name__ == "__main__":
    pytest.main()
