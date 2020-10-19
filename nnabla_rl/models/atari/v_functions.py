import nnabla as nn

import nnabla.parametric_functions as PF

import nnabla_rl.initializers as RI
from nnabla_rl.models.v_function import VFunction


class PPOVFunction(VFunction):
    """
    Shared parameter function proposed used in PPO paper for atari environment.
    This network outputs the value
    See: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, head, scope_name, state_shape):
        super(PPOVFunction, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._head = head

    def v(self, s):
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = PF.affine(h, n_outmaps=1,
                              w_init=RI.NormcInitializer(std=0.01))
        return v

    def _hidden(self, s):
        assert s.shape[1:] == self._state_shape
        return self._head(s)
