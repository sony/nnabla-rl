import nnabla as nn

import nnabla.functions as NF
import nnabla.parametric_functions as NPF

import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import StochasticPolicy


class PPOPolicy(StochasticPolicy):
    """
    Shared parameter function proposed used in PPO paper for atari environment.
    This network outputs the policy distribution.
    See: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, head, scope_name, state_shape, action_dim):
        super(PPOPolicy, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._action_dim = action_dim
        self._head = head

    def pi(self, s):
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_pi"):
                z = NPF.affine(h, n_outmaps=self._action_dim,
                               w_init=RI.NormcInitializer(std=0.01))
        return D.Softmax(z=z)

    def _hidden(self, s):
        assert s.shape[1:] == self._state_shape
        return self._head(s)


class ICML2015TRPOPolicy(StochasticPolicy):
    """
    Policy network proposed in TRPO original paper for atari environment.
    This network outputs the value and policy distribution.
    See: https://arxiv.org/pdf/1502.05477.pdf
    """

    def __init__(self, scope_name, state_shape, action_dim):
        super(ICML2015TRPOPolicy, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._action_dim = action_dim

    def pi(self, s):
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NF.tanh(NPF.convolution(
                    s, 16, (4, 4), stride=(2, 2)))
            with nn.parameter_scope("conv2"):
                h = NF.tanh(NPF.convolution(
                    h, 16, (4, 4), pad=(1, 1), stride=(2, 2)))
            h = NF.reshape(h, (batch_size, -1), inplace=False)
            with nn.parameter_scope("affine1"):
                h = NF.tanh(NPF.affine(h, 20))
            with nn.parameter_scope("affine2"):
                z = NPF.affine(h, self._action_dim)

        return D.Softmax(z=z)
