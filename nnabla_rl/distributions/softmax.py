# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.distributions import Distribution


class Softmax(Distribution):
    '''
    Softmax distribution which samples a class index :math:`i` according to the following probability.

    :math:`i \\sim \\frac{\\exp{z_{i}}}{\\sum_{j}\\exp{z_{j}}}`.

    Args:
        z (nn.Variable): logits :math:`z`. Logits' dimension should be same as the number of class to sample.
    '''

    def __init__(self, z):
        super(Softmax, self).__init__()
        if not isinstance(z, nn.Variable):
            z = nn.Variable.from_numpy_array(z)

        self._distribution = NF.softmax(x=z, axis=len(z.shape) - 1)
        self._log_distribution = NF.log_softmax(x=z, axis=len(z.shape) - 1)
        self._batch_size = z.shape[0]
        self._num_class = z.shape[-1]

        labels = np.array(
            [label for label in range(self._num_class)], dtype=np.int)
        self._labels = nn.Variable.from_numpy_array(labels)
        self._actions = self._labels
        for size in reversed(z.shape[0:-1]):
            self._actions = NF.stack(*[self._actions for _ in range(size)])

    @property
    def ndim(self):
        return 1

    def sample(self, noise_clip=None):
        # NOTE: nnabla's random_choice backpropagetes through distribution
        return NF.random_choice(self._actions, w=self._distribution)

    def sample_multiple(self, num_samples, noise_clip=None):
        raise NotImplementedError

    def sample_and_compute_log_prob(self, noise_clip=None):
        # NOTE: nnabla's random_choice backpropagetes through distribution
        sample = NF.random_choice(self._actions, w=self._distribution)
        log_prob = self.log_prob(sample)
        return sample, log_prob

    def choose_probable(self):
        # NOTE: nnabla's argmax backpropagetes through distribution
        return RF.argmax(self._distribution, axis=len(self._distribution.shape) - 1)

    def mean(self):
        raise NotImplementedError

    def log_prob(self, x):
        one_hot_action = NF.one_hot(x, shape=(self._num_class, ))
        return NF.sum(self._log_distribution * one_hot_action, axis=len(self._distribution.shape) - 1, keepdims=True)

    def entropy(self):
        plogp = self._distribution * self._log_distribution
        return -NF.sum(plogp, axis=len(plogp.shape) - 1, keepdims=True)

    def kl_divergence(self, q):
        if not isinstance(q, Softmax):
            raise ValueError("Invalid q to compute kl divergence")
        return NF.sum(self._distribution * (self._log_distribution - q._log_distribution),
                      axis=len(self._distribution.shape) - 1,
                      keepdims=True)
