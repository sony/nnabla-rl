import nnabla as nn
import nnabla.functions as NF

import numpy as np

from nnabla_rl.distributions import Distribution
import nnabla_rl.functions as RF


class Softmax(Distribution):
    '''
    Softmax distribution
    '''

    def __init__(self, z):
        super(Softmax, self).__init__()
        if not isinstance(z, nn.Variable):
            z = nn.Variable.from_numpy_array(z)

        self._distribution = NF.softmax(x=z, axis=1)
        self._batch_size = z.shape[0]
        self._num_class = z.shape[-1]

        labels = np.array(
            [label for label in range(self._num_class)], dtype=np.int)
        self._labels = nn.Variable.from_numpy_array(labels)
        self._actions = NF.stack(
            *[self._labels for _ in range(self._batch_size)])

    def sample(self, noise_clip=None):
        return RF.random_choice(self._actions, w=self._distribution)

    def sample_multiple(self, num_samples, noise_clip=None):
        raise NotImplementedError

    def sample_and_compute_log_prob(self, noise_clip=None):
        sample = RF.random_choice(self._actions, w=self._distribution)
        log_prob = self.log_prob(sample)
        return sample, log_prob

    def choose_probable(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def log_prob(self, x):
        log_pi = NF.log(self._distribution)
        one_hot_action = NF.one_hot(x, shape=(self._num_class, ))
        return NF.sum(log_pi * one_hot_action, axis=1, keepdims=True)

    def entropy(self):
        plogp = self._distribution * NF.log(self._distribution)
        return -NF.sum(plogp, axis=1, keepdims=True)

    def kl_divergence(self, q):
        if not isinstance(q, Softmax):
            raise ValueError("Invalid q to compute kl divergence")
        return NF.sum(self._distribution * (NF.log(self._distribution) - NF.log(q._distribution)),
                      axis=1,
                      keepdims=True)
