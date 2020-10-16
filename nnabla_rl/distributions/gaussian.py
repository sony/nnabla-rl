import nnabla as nn
import nnabla.functions as F

import numpy as np

import nnabla_rl.functions as RF
from nnabla_rl.distributions import Distribution, common_utils


class Gaussian(Distribution):
    '''
    Gaussian distribution
    '''

    def __init__(self, mean, ln_var):
        super(Gaussian, self).__init__()
        if not isinstance(mean, nn.Variable):
            mean = nn.Variable.from_numpy_array(mean)
        if not isinstance(ln_var, nn.Variable):
            ln_var = nn.Variable.from_numpy_array(ln_var)

        self._mean = mean
        self._var = F.exp(ln_var)
        self._ln_var = ln_var
        self._batch_size = mean.shape[0]
        self._data_dim = mean.shape[1:]

    def sample(self, noise_clip=None):
        return RF.sample_gaussian(self._mean,
                                  self._ln_var,
                                  noise_clip=noise_clip)

    def sample_multiple(self, num_samples, noise_clip=None):
        return RF.sample_gaussian_multiple(self._mean,
                                           self._ln_var,
                                           num_samples,
                                           noise_clip=noise_clip)

    def sample_and_compute_log_prob(self, noise_clip=None):
        x = RF.sample_gaussian(mean=self._mean,
                               ln_var=self._ln_var,
                               noise_clip=noise_clip)
        return x, self.log_prob(x)

    def sample_multiple_and_compute_log_prob(self, num_samples, noise_clip=None):
        x = RF.sample_gaussian_multiple(self._mean,
                                        self._ln_var,
                                        num_samples,
                                        noise_clip=noise_clip)
        mean = RF.expand_dims(self._mean, axis=1)
        var = RF.expand_dims(self._var, axis=1)
        ln_var = RF.expand_dims(self._ln_var, axis=1)

        assert mean.shape == (self._batch_size, 1, ) + self._data_dim
        assert var.shape == mean.shape
        assert ln_var.shape == mean.shape

        return x, common_utils.gaussian_log_prob(x, mean, var, ln_var)

    def choose_probable(self):
        return self._mean

    def mean(self):
        return self._mean

    def log_prob(self, x):
        return common_utils.gaussian_log_prob(x, self._mean, self._var, self._ln_var)

    def entropy(self):
        return F.sum(0.5 + 0.5 * np.log(2.0 * np.pi) + 0.5 * self._ln_var, axis=1, keepdims=True)

    def kl_divergence(self, q):
        assert isinstance(q, Gaussian)
        p = self
        return 0.5 * F.sum(q._ln_var - p._ln_var + (p._var + (p._mean - q._mean) ** 2.0) / q._var - 1, axis=1, keepdims=True)
