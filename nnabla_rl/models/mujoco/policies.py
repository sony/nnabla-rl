import numpy as np

import nnabla as nn

import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import nnabla.initializer as I
from nnabla.parameter import get_parameter_or_create

import numpy as np

import nnabla_rl.distributions as D
import nnabla_rl.initializers as RI
from nnabla_rl.models.policy import DeterministicPolicy, StochasticPolicy, preprocess_state


class TD3Policy(DeterministicPolicy):
    """
    Actor model proposed by S. Fujimoto in TD3 paper for mujoco environment.
    See: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, scope_name, state_dim, action_dim, max_action_value):
        super(TD3Policy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            linear1_init = RI.HeUniform(
                inmaps=self._state_dim, outmaps=400, factor=1/3)
            h = PF.affine(s, n_outmaps=400, name="linear1",
                          w_init=linear1_init, b_init=linear1_init)
            h = F.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = PF.affine(h, n_outmaps=300, name="linear2",
                          w_init=linear2_init, b_init=linear2_init)
            h = F.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=self._action_dim, factor=1/3)
            h = PF.affine(h, n_outmaps=self._action_dim, name="linear3",
                          w_init=linear3_init, b_init=linear3_init)
        return F.tanh(h) * self._max_action_value


class SACPolicy(StochasticPolicy):
    """
    Actor model proposed by T. Haarnoja in SAC paper for mujoco environment.
    See: https://arxiv.org/pdf/1801.01290.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim,
                 clip_log_sigma=True, min_log_sigma=-20.0, max_log_sigma=2):
        super(SACPolicy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._clip_log_sigma = clip_log_sigma
        self._min_log_sigma = min_log_sigma
        self._max_log_sigma = max_log_sigma

    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=256, name="linear1")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=256, name="linear2")
            h = F.relu(x=h)
            h = PF.affine(h, n_outmaps=self._action_dim*2, name="linear3")
            reshaped = F.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_sigma = F.split(reshaped, axis=1)
            assert mean.shape == ln_sigma.shape
            assert mean.shape == (s.shape[0], self._action_dim)
            if self._clip_log_sigma:
                ln_sigma = F.clip_by_value(
                    ln_sigma, min=self._min_log_sigma, max=self._max_log_sigma)
            ln_var = ln_sigma * 2.0
        return D.SquashedGaussian(mean=mean, ln_var=ln_var)


class BEARPolicy(StochasticPolicy):
    """
    Actor model proposed by A. Kumar, et al. in BEAR paper for mujoco environment.
    See: https://arxiv.org/pdf/1906.00949.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(BEARPolicy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            linear1_init = RI.HeUniform(
                inmaps=self._state_dim, outmaps=400, factor=1/3)
            h = PF.affine(s, n_outmaps=400, name="linear1",
                          w_init=linear1_init, b_init=linear1_init)
            h = F.relu(x=h)
            linear2_init = RI.HeUniform(
                inmaps=400, outmaps=300, factor=1/3)
            h = PF.affine(h, n_outmaps=300, name="linear2",
                          w_init=linear2_init, b_init=linear2_init)
            h = F.relu(x=h)
            linear3_init = RI.HeUniform(
                inmaps=300, outmaps=self._action_dim*2, factor=1/3)
            h = PF.affine(h, n_outmaps=self._action_dim*2, name="linear3",
                          w_init=linear3_init, b_init=linear3_init)
            reshaped = F.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_var = F.split(reshaped, axis=1)
            assert mean.shape == ln_var.shape
            assert mean.shape == (s.shape[0], self._action_dim)
        return D.Gaussian(mean=mean, ln_var=ln_var)


class PPOPolicy(StochasticPolicy):
    """
    Actor model proposed by John Schulman, et al. in PPO paper for mujoco environment.
    This network outputs the policy distribution
    See: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(PPOPolicy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    @preprocess_state
    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=64, name="linear1",
                          w_init=RI.NormcInitializer(std=1.0))
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=64, name="linear2",
                          w_init=RI.NormcInitializer(std=1.0))
            h = F.tanh(x=h)
            mean = PF.affine(h, n_outmaps=self._action_dim, name="linear3",
                          w_init=RI.NormcInitializer(std=0.01))
            ln_sigma = nn.parameter.get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=I.ConstantInitializer(0.))
            ln_var = F.broadcast(ln_sigma, (s.shape[0], self._action_dim)) * 2.0
            assert mean.shape == ln_var.shape
            assert mean.shape == (s.shape[0], self._action_dim)
        return D.Gaussian(mean=mean, ln_var=ln_var)


class ICML2015TRPOPolicy(StochasticPolicy):
    """
    Actor model proposed by John Schulman, et al. in TRPO paper for mujoco environment.
    See: https://arxiv.org/pdf/1502.05477.pdf (Original paper)
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(ICML2015TRPOPolicy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=30, name="linear1")
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=self._action_dim*2, name="linear2")
            reshaped = F.reshape(
                h, shape=(-1, 2, self._action_dim), inplace=False)
            mean, ln_sigma = F.split(reshaped, axis=1)
            assert mean.shape == ln_sigma.shape
            assert mean.shape == (s.shape[0], self._action_dim)
            ln_var = ln_sigma * 2.0
        return D.Gaussian(mean=mean, ln_var=ln_var)


class TRPOPolicy(StochasticPolicy):
    """
    Actor model proposed by Peter Henderson, et al.
    in Deep Reinforcement Learning that Matters paper for mujoco environment.
    See: https://arxiv.org/abs/1709.06560.pdf
    """

    def __init__(self, scope_name, state_dim, action_dim):
        super(TRPOPolicy, self).__init__(scope_name)
        self._state_dim = state_dim
        self._action_dim = action_dim

    @preprocess_state
    def pi(self, s):
        assert s.shape[1] == self._state_dim

        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=64, name="linear1",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.)))
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=64, name="linear2",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.)))
            h = F.tanh(x=h)
            mean = PF.affine(h, n_outmaps=self._action_dim, name="linear3",
                             w_init=I.OrthogonalInitializer(np.sqrt(2.)))
            assert mean.shape == (s.shape[0], self._action_dim)

            ln_sigma = get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=I.ConstantInitializer(0.))
            ln_var = F.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
        return D.Gaussian(mean, ln_var)
