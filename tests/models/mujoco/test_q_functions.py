import pytest

import nnabla as nn
import numpy as np

from nnabla_rl.models.mujoco.q_functions import TD3QFunction


class TestTD3QFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = TD3QFunction(scope_name=scope_name,
                             state_dim=state_dim,
                             action_dim=action_dim)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_dim = 5
        action_dim = 5
        scope_name = "test"
        model = TD3QFunction(scope_name=scope_name,
                             state_dim=state_dim,
                             action_dim=action_dim)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, state_dim))
        input_action = nn.Variable.from_numpy_array(np.ones((1, action_dim)))
        model.q(input_state, input_action)

        assert len(model.get_parameters()) == 6


if __name__ == "__main__":
    pytest.main()
