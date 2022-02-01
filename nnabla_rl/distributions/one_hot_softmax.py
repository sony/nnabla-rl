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

import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.distributions.softmax import Softmax


class OneHotSoftmax(Softmax):
    '''
    Softmax distribution which samples a one-hot vector of class index :math:`i` as 1.
    Class index is sampled according to the following distribution.

    :math:`i \\sim \\frac{\\exp{z_{i}}}{\\sum_{j}\\exp{z_{j}}}`.

    Args:
        z (nn.Variable): logits :math:`z`. Logits' dimension should be same as the number of class to sample.
    '''

    def __init__(self, z):
        super(OneHotSoftmax, self).__init__(z)

    @property
    def ndim(self):
        return 1

    def sample(self, noise_clip=None):
        sample = NF.random_choice(self._actions, w=self._distribution)
        one_hot = NF.one_hot(sample, shape=(self._num_class, ))
        one_hot.need_grad = False
        # straight through biased gradient estimator
        assert one_hot.shape == self._distribution.shape
        one_hot = one_hot + (self._distribution - self._distribution.get_unlinked_variable(need_grad=False))
        return one_hot

    def sample_and_compute_log_prob(self, noise_clip=None):
        sample = NF.random_choice(self._actions, w=self._distribution)
        log_prob = self.log_prob(sample)
        one_hot = NF.one_hot(sample, shape=(self._num_class, ))
        one_hot.need_grad = False
        # straight through biased gradient estimator
        assert one_hot.shape == self._distribution.shape
        one_hot = one_hot + (self._distribution - self._distribution.get_unlinked_variable(need_grad=False))
        return one_hot, log_prob

    def choose_probable(self):
        class_index = RF.argmax(self._distribution, axis=len(self._distribution.shape) - 1, keepdims=True)
        one_hot = NF.one_hot(class_index, shape=(self._num_class, ))
        one_hot.need_grad = False
        # straight through biased gradient estimator
        assert one_hot.shape == self._distribution.shape
        one_hot = one_hot + (self._distribution - self._distribution.get_unlinked_variable(need_grad=False))
        return one_hot

    def log_prob(self, x):
        return NF.sum(self._log_distribution * x, axis=len(x.shape) - 1, keepdims=True)
