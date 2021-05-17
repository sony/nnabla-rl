# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import nnabla.initializer as NI
from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors.preprocessor import Preprocessor


class RunningMeanNormalizer(Preprocessor, Model):
    def __init__(self, scope_name, shape, epsilon=1e-8, value_clip=None):
        super(RunningMeanNormalizer, self).__init__(scope_name)

        if value_clip is not None and value_clip[0] > value_clip[1]:
            raise ValueError(
                f'Unexpected clipping value range: {value_clip[0]} > {value_clip[1]}')
        self._value_clip = value_clip
        self._epsilon = epsilon
        self._shape = shape

    def process(self, x):
        assert 0 < self._count.d
        std = (self._var + self._epsilon) ** 0.5
        normalized = (x - self._mean) / std
        if self._value_clip is not None:
            normalized = NF.clip_by_value(
                normalized, min=self._value_clip[0], max=self._value_clip[1])
        normalized.need_grad = False
        return normalized

    def update(self, data):
        avg_a = self._mean.d
        var_a = self._var.d
        n_a = self._count.d

        avg_b = np.mean(data, axis=0)
        var_b = np.var(data, axis=0)
        n_b = data.shape[0]

        n_ab = n_a + n_b
        delta = avg_b - avg_a

        self._mean.d = avg_a + delta * n_b / n_ab
        m_a = var_a * n_a
        m_b = var_b * n_b
        M2 = m_a + m_b + np.square(delta) * n_a * n_b / n_ab
        self._var.d = M2 / n_ab
        self._count.d = n_ab

    @property
    def _mean(self):
        with nn.parameter_scope(self.scope_name):
            return nn.parameter.get_parameter_or_create(name='mean', shape=(1, *self._shape),
                                                        initializer=NI.ConstantInitializer(0.0))

    @property
    def _var(self):
        with nn.parameter_scope(self.scope_name):
            return nn.parameter.get_parameter_or_create(name='var', shape=(1, *self._shape),
                                                             initializer=NI.ConstantInitializer(1.0))

    @property
    def _count(self):
        with nn.parameter_scope(self.scope_name):
            return nn.parameter.get_parameter_or_create(name='count', shape=(1, 1),
                                                        initializer=NI.ConstantInitializer(1e-4))
