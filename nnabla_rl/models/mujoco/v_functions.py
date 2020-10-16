import numpy as np

import nnabla as nn

import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from nnabla_rl.models.v_function import VFunction, preprocess_state
import nnabla_rl.initializers as RI

class SACVFunction(VFunction):
    """
    VFunciton model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    """

    def __init__(self, scope_name, state_dim):
        super(SACVFunction, self).__init__(scope_name)
        self._state_dim = state_dim

    def v(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=256, name="linear1")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=256, name="linear2")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=1, name="linear3")
        return h


class TRPOVFunction(VFunction):
    """
    Vfunction proposed by Peter Henderson, et al.
    in Deep Reinforcement Learning that Matters paper for mujoco environment.
    See: https://arxiv.org/abs/1709.06560.pdf
    """

    def __init__(self, scope_name, state_dim):
        super(TRPOVFunction, self).__init__(scope_name)
        self._state_dim = state_dim

    @preprocess_state
    def v(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=64, name="linear1",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.)))
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=64, name="linear2",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.)))
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=1, name="linear3",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.)))
        return h


class PPOVFunction(VFunction):
    """
    Shared parameter function proposed used in PPO paper for mujoco environment.
    This network outputs the state value
    See: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, scope_name, state_shape):
        super(PPOVFunction, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape

    @preprocess_state
    def v(self, s):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear1"):
                h = PF.affine(s, n_outmaps=64,
                              w_init=RI.NormcInitializer(std=1.0))
            h = F.tanh(x=h)
            with nn.parameter_scope("linear2"):
                h = PF.affine(h, n_outmaps=64,
                              w_init=RI.NormcInitializer(std=1.0))
            h = F.tanh(x=h)
            with nn.parameter_scope("linear_v"):
                v = PF.affine(h, n_outmaps=1,
                              w_init=RI.NormcInitializer(std=1.0))
        return v
