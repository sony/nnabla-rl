import nnabla as nn

import nnabla.functions as NF
import nnabla.parametric_functions as NPF

import nnabla_rl.distributions as D
import nnabla_rl.functions as RF
from nnabla_rl.models.variational_auto_encoder import VariationalAutoEncoder


class UnsquashedVariationalAutoEncoder(VariationalAutoEncoder):
    """
    Almost identical to BCQ style variational auto encoder proposed by S. Fujimoto in BCQ paper for mujoco environment.
    See: https://arxiv.org/pdf/1812.02900.pdf
    The main difference is that the output action is not squashed with tanh for computational convenience.
    """

    def __init__(self, scope_name, state_dim, action_dim, latent_dim):
        super(UnsquashedVariationalAutoEncoder, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._latent_dim = latent_dim

    def __call__(self, *args):
        assert len(args) == 2
        (s, a) = args
        latent_distribution = self.latent_distribution(*args)
        z = latent_distribution.sample()
        reconstructed = self.decode(s, z)
        return latent_distribution, reconstructed

    def latent_distribution(self, *args):
        assert len(args) == 2
        x = NF.concatenate(*args)
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(x, n_outmaps=750, name="linear1")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear2")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._latent_dim*2, name="linear3")
            reshaped = NF.reshape(h, shape=(-1, 2, self._latent_dim))
            mean, ln_var = NF.split(reshaped, axis=1)
            # Clip for numerical stability
            ln_var = NF.clip_by_value(ln_var, min=-8, max=30)
        return D.Gaussian(mean, ln_var)

    def encode(self, *args):
        assert len(args) == 2
        x = NF.concatenate(*args)
        latent_distribution = self.latent_distribution(x)
        return latent_distribution.sample()

    def decode(self, *args):
        assert (len(args) == 1) or (len(args) == 2)
        if len(args) == 1:
            s, *_ = args
            z = NF.randn(shape=(s.shape[0], self._latent_dim))
            z = NF.clip_by_value(z, -0.5, 0.5)
        else:
            (s, z) = args
        with nn.parameter_scope(self.scope_name):
            x = NF.concatenate(s, z)
            h = NPF.affine(x, n_outmaps=750, name="linear4")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear5")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear6")
        return h

    def decode_multiple(self, decode_num, *args):
        assert (len(args) == 1) or (len(args) == 2)
        if len(args) == 1:
            (s, ) = args
            z = NF.randn(shape=(s.shape[0], decode_num, self._latent_dim))
            z = NF.clip_by_value(z, -0.5, 0.5)
        else:
            (s, z) = args
        s = RF.expand_dims(s, axis=0)
        s = RF.repeat(s, repeats=decode_num, axis=0)
        s = NF.transpose(s, axes=(1, 0, 2))
        assert s.shape[:-1] == z.shape[:-1]

        x = NF.concatenate(s, z, axis=2)
        x = NF.reshape(x, shape=(-1, x.shape[-1]))
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(x, n_outmaps=750, name="linear4")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=750, name="linear5")
            h = NF.relu(x=h)
            h = NPF.affine(h, n_outmaps=self._action_dim, name="linear6")
            h = NF.reshape(h, shape=(-1, decode_num, h.shape[-1]))
        return h


class BCQVariationalAutoEncoder(UnsquashedVariationalAutoEncoder):
    """
    BCQ style variational auto encoder proposed by S. Fujimoto in BCQ paper for mujoco environment.
    See: https://arxiv.org/pdf/1812.02900.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim, latent_dim, max_action_value):
        super(BCQVariationalAutoEncoder, self).__init__(
            scope_name, state_dim, action_dim, latent_dim)
        self._max_action_value = max_action_value

    def decode(self, *args):
        unsquashed = super(BCQVariationalAutoEncoder, self).decode(*args)
        return NF.tanh(unsquashed) * self._max_action_value

    def decode_multiple(self, decode_num, *args):
        unsquashed = super(BCQVariationalAutoEncoder,
                           self).decode_multiple(decode_num, *args)
        return NF.tanh(unsquashed) * self._max_action_value
