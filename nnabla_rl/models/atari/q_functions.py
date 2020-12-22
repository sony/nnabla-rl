import nnabla as nn

import nnabla.functions as NF
import nnabla.parametric_functions as NPF

import nnabla_rl.initializers as RI
from nnabla_rl.models.q_function import QFunction


class DQNQFunction(QFunction):
    """
    Q function proposed by DeepMind in DQN paper for atari environment.
    See: https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning
    """

    _n_action: int

    def __init__(self, scope_name: str, n_action: int):
        super(DQNQFunction, self).__init__(scope_name)
        self._n_action = n_action

    def _predict_q_values(self, s: nn.Variable) -> nn.Variable:
        """ Predict all q values of the given state
        """
        with nn.parameter_scope(self.scope_name):

            with nn.parameter_scope("conv1"):
                h = NF.relu(NPF.convolution(s, 32, (8, 8), stride=(4, 4),
                                            w_init=RI.HeNormal(s.shape[1],
                                                               32,
                                                               kernel=(8, 8))
                                            ))

            with nn.parameter_scope("conv2"):
                h = NF.relu(NPF.convolution(h, 64, (4, 4), stride=(2, 2),
                                            w_init=RI.HeNormal(h.shape[1],
                                                               64,
                                                               kernel=(4, 4))
                                            ))

            with nn.parameter_scope("conv3"):
                h = NF.relu(NPF.convolution(h, 64, (3, 3), stride=(1, 1),
                                            w_init=RI.HeNormal(h.shape[1],
                                                               64,
                                                               kernel=(3, 3))
                                            ))

            h = NF.reshape(h, (-1, 3136))

            with nn.parameter_scope("affine1"):
                h = NF.relu(NPF.affine(h, 512,
                                       w_init=RI.HeNormal(h.shape[1], 512)
                                       ))

            with nn.parameter_scope("affine2"):
                h = NPF.affine(h, self._n_action,
                               w_init=RI.HeNormal(h.shape[1], self._n_action)
                               )
        return h

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        q_values = self._predict_q_values(s)

        q_value = NF.sum(q_values
                         * NF.one_hot(NF.reshape(a, (-1, 1), inplace=False),
                                      (q_values.shape[1],)),
                         axis=1, keepdims=True)  # get q value of a

        return q_value

    def max_q(self, s: nn.Variable) -> nn.Variable:
        q_values = self._predict_q_values(s)
        return NF.max(q_values, axis=1, keepdims=True)
