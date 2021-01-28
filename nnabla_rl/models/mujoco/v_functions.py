import numpy as np

import nnabla as nn

import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla.initializer as NI

from nnabla_rl.models.v_function import VFunction
import nnabla_rl.initializers as RI


class SACVFunction(VFunction):
    """
    VFunciton model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    """

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=256, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=256, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3")
        return h


class TRPOVFunction(VFunction):
    """
    Vfunction proposed by Peter Henderson, et al.
    in Deep Reinforcement Learning that Matters paper for mujoco environment.
    See: https://arxiv.org/abs/1709.06560.pdf
    """

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=64, name="linear1",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=64, name="linear2",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3",
                           w_init=NI.OrthogonalInitializer(np.sqrt(2.)))
        return h


class PPOVFunction(VFunction):
    """
    Shared parameter function proposed used in PPO paper for mujoco environment.
    This network outputs the state value
    See: https://arxiv.org/pdf/1707.06347.pdf
    """

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear1"):
                h = NPF.affine(s, n_outmaps=64,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            with nn.parameter_scope("linear2"):
                h = NPF.affine(h, n_outmaps=64,
                               w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1,
                               w_init=RI.NormcInitializer(std=1.0))
        return v


class GAILVFunction(VFunction):
    """
    Value function proposed by Jonathan Ho, et al.
    See: https://arxiv.org/pdf/1606.03476.pdf
    """

    def __init__(self, scope_name: str):
        super(GAILVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, n_outmaps=100, name="linear1",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=100, name="linear2",
                           w_init=RI.NormcInitializer(std=1.0))
            h = NF.tanh(x=h)
            h = NPF.affine(h, n_outmaps=1, name="linear3",
                           w_init=RI.NormcInitializer(std=1.0))
        return h
