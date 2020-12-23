import numpy as np
import nnabla as nn

from nnabla_rl.models.classic_control.policies import REINFORCEContinousPolicy, REINFORCEDiscretePolicy


class TestREINFORCEContinousPolicy(object):
    def test_scope_name(self):
        nn.clear_parameters()

        action_dim = 5
        scope_name = "test"
        fixed_ln_var = np.exp(0.1)

        model = REINFORCEContinousPolicy(scope_name, action_dim, fixed_ln_var)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_dim = 5
        action_dim = 5
        scope_name = "test"
        fixed_ln_var = np.exp(0.1)

        model = REINFORCEContinousPolicy(scope_name, action_dim, fixed_ln_var)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, state_dim))
        model.pi(input_state)

        assert scope_name == model.scope_name

        assert len(model.get_parameters()) == 6


class TestREINFORCEDiscretePolicy(object):
    def test_scope_name(self):
        nn.clear_parameters()

        action_dim = 5
        scope_name = "test"

        model = REINFORCEDiscretePolicy(scope_name, action_dim)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_dim = 5
        action_dim = 5
        scope_name = "test"

        model = REINFORCEDiscretePolicy(scope_name, action_dim)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, state_dim))
        model.pi(input_state)

        assert scope_name == model.scope_name

        assert len(model.get_parameters()) == 6
