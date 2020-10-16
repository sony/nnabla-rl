import numpy as np
import nnabla as nn

import nnabla.functions as F
import nnabla.parametric_functions as PF

import nnabla_rl.distributions as D
from nnabla_rl.models.policy import StochasticPolicy
import nnabla_rl.initializers as RI


class REINFORCEDiscretePolicy(StochasticPolicy):
    """
    REINFORCE policy for classic control discrete environment.
    This network outputs the policy distribution.
    See: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(REINFORCEDiscretePolicy, self).__init__(scope_name=scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

        dummy_state = nn.Variable((1, state_dim))

        # Dummy call. Just for initializing the parameters
        self.pi(dummy_state)

    def pi(self, s):
        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=200, name="linear1",
                          w_init=RI.HeNormal(s.shape[1], 200))
            h = F.leaky_relu(h)
            h = PF.affine(h, n_outmaps=200, name="linear2",
                          w_init=RI.HeNormal(s.shape[1], 200))
            h = F.leaky_relu(h)
            z = PF.affine(h, n_outmaps=self._action_dim,
                          name="linear3", w_init=RI.LeCunNormal(s.shape[1], 200))

        return D.Softmax(z=z)


class REINFORCEContinousPolicy(StochasticPolicy):
    """
    REINFORCE policy for classic control continous environment.
    This network outputs the policy distribution.
    See: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim, fixed_ln_var):
        super(REINFORCEContinousPolicy, self).__init__(scope_name=scope_name)
        self._state_shape = state_dim
        self._action_dim = action_dim
        if np.isscalar(fixed_ln_var):
            self._fixed_ln_var = np.full(self._action_dim, fixed_ln_var)
        else:
            self._fixed_ln_var = fixed_ln_var

        dummy_state = nn.Variable((1, state_dim))

        # Dummy call. Just for initializing the parameters
        self.pi(dummy_state)

    def pi(self, s):
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=200, name="linear1",
                          w_init=RI.HeNormal(s.shape[1], 200))
            h = F.leaky_relu(h)
            h = PF.affine(h, n_outmaps=200, name="linear2",
                          w_init=RI.HeNormal(s.shape[1], 200))
            h = F.leaky_relu(h)
            z = PF.affine(h, n_outmaps=self._action_dim,
                          name="linear3", w_init=RI.HeNormal(s.shape[1], 200))

        return D.Gaussian(z, np.tile(self._fixed_ln_var, (batch_size, 1)))
