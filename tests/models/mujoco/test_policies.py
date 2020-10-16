import pytest

import nnabla as nn
import numpy as np

from nnabla_rl.models.mujoco.policies import TD3Policy, SACPolicy


class TestTD3Policy(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_scope_name(self):
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = TD3Policy(scope_name=scope_name,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          max_action_value=1.0)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = TD3Policy(scope_name=scope_name,
                          state_dim=state_dim,
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
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = SACPolicy(scope_name=scope_name,
                          state_dim=state_dim,
                          action_dim=action_dim)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = SACPolicy(scope_name=scope_name,
                          state_dim=state_dim,
                          action_dim=action_dim)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, state_dim))
        model.pi(input_state)

        assert len(model.get_parameters()) == 6


if __name__ == "__main__":
    pytest.main()
