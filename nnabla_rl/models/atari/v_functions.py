import nnabla as nn
import nnabla.parametric_functions as NPF

import nnabla_rl.initializers as RI
from nnabla_rl.models.v_function import VFunction
from nnabla_rl.models.atari.shared_functions import PPOSharedFunctionHead


class PPOVFunction(VFunction):
    '''
    Shared parameter function proposed used in PPO paper for atari environment.
    This network outputs the value
    See: https://arxiv.org/pdf/1707.06347.pdf
    '''

    _head: PPOSharedFunctionHead

    def __init__(self, head: PPOSharedFunctionHead, scope_name: str):
        super(PPOVFunction, self).__init__(scope_name=scope_name)
        self._head = head

    def v(self, s: nn.Variable) -> nn.Variable:
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1,
                               w_init=RI.NormcInitializer(std=0.01))
        return v

    def _hidden(self, s: nn.Variable) -> nn.Variable:
        return self._head(s)


class A3CVFunction(VFunction):
    '''
    Shared parameter function proposed and used in A3C paper for atari environment.
    See: https://arxiv.org/pdf/1602.01783.pdf
    '''

    def __init__(self, head, scope_name, state_shape):
        super(A3CVFunction, self).__init__(scope_name=scope_name)
        self._state_shape = state_shape
        self._head = head

    def v(self, s):
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = NPF.affine(h, n_outmaps=1)
        return v

    def _hidden(self, s):
        assert s.shape[1:] == self._state_shape
        return self._head(s)
