import nnabla as nn

import nnabla.functions as F
import nnabla.parametric_functions as PF

import nnabla_rl.initializers as RI
from nnabla_rl.models.q_function import QFunction


class TD3QFunction(QFunction):
    """
    Critic model proposed by S. Fujimoto in TD3 paper for mujoco environment.
    See: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(TD3QFunction, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def q(self, s, a):
        assert s.shape[1] == self._state_dim
        assert a.shape[1] == self._action_dim

        with nn.parameter_scope(self.scope_name):
            h = F.concatenate(s, a)
            linear1_init = RI.HeUniform(
                inmaps=h.shape[1], outmaps=400, factor=1/3)
            h = PF.affine(h, n_outmaps=400, name="linear1",
                          w_init=linear1_init, b_init=linear1_init)
            h = F.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = PF.affine(h, n_outmaps=300, name="linear2",
                          w_init=linear2_init, b_init=linear2_init)
            h = F.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=1, factor=1/3)
            h = PF.affine(h, n_outmaps=1, name="linear3",
                          w_init=linear3_init, b_init=linear3_init)
        return h


class SACQFunction(QFunction):
    """
    QFunciton model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(SACQFunction, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def q(self, s, a):
        assert s.shape[1] == self._state_dim
        assert a.shape[1] == self._action_dim

        with nn.parameter_scope(self.scope_name):
            h = F.concatenate(s, a)
            h = PF.affine(h, n_outmaps=256, name="linear1")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=256, name="linear2")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=1, name="linear3")
        return h
