# Copyright 2022 Sony Group Corporation.
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


class Bernoulli(Distribution):
    '''
    Bernoulli distribution.

    :math:`p^{k}(1-p)^{1-k} \\enspace \\text{for}\\ k\\in\\{0,1\\}`.

    Args:
        z (nn.Variable): Probability of outputting 1 is computed as :math:`p=sigmoid(z)`.
    '''

    def __init__(self, z):
        super(Bernoulli, self).__init__()
        assert z.shape[-1] == 1

        logit = nn.Variable.from_numpy_array(z) if not isinstance(z, nn.Variable) else z
        self._logit = logit
        self._p = NF.sigmoid(logit)
        self._log_p = NF.softplus(logit, beta=-1.0)
        self._log_1_minus_p = -logit + NF.softplus(logit, beta=-1.0)
        self._distribution = NF.concatenate(self._p, 1 - self._p)
        self._log_distribution = NF.concatenate(self._log_p, self._log_1_minus_p)

        labels = np.array([1, 0], dtype=np.int)
        labels = nn.Variable.from_numpy_array(labels)
        self._labels = labels
        for size in reversed(z.shape[0:-1]):
            self._labels = NF.stack(*[self._labels for _ in range(size)])

    @property
    def ndim(self):
        return 1

    def sample(self, noise_clip=None):
        '''
        Sample a value from the distribution.

        Args:
            noise_clip(Tuple[float, float], optional): Noise clip does nothing in Bernoulli distribution.

        Returns:
            nn.Variable: Sampled value.
        '''
        return NF.random_choice(self._labels, w=self._distribution)

    def sample_and_compute_log_prob(self, noise_clip=None):
        '''
        Sample a value from the distribution and compute its log probability.

        Args:
            noise_clip(Tuple[float, float], optional): Noise clip does nothing in Bernoulli distribution.

        Returns:
            Tuple[nn.Variable, nn.Variable]: Sampled value and its log probabilty
        '''
        x = self.sample(noise_clip=noise_clip)
        return x, self.log_prob(x)

    def choose_probable(self):
        # NOTE: nnabla's argmax backpropagetes through distribution
        return RF.argmax(self._distribution, axis=len(self._distribution.shape) - 1)

    def mean(self):
        return self._p

    def log_prob(self, x):
        return -NF.sigmoid_cross_entropy(self._logit, x)

    def entropy(self):
        return (1 - NF.sigmoid(self._logit)) * self._logit - NF.log_sigmoid(self._logit)

    def kl_divergence(self, q):
        assert isinstance(q, Bernoulli)
        return NF.sum(self._distribution * (self._log_distribution - q._log_distribution),
                      axis=len(self._distribution.shape) - 1,
                      keepdims=True)
