import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.initializer as I

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

        with nn.parameter_scope(scope_name):
            self._mean = \
                nn.parameter.get_parameter_or_create(name='mean',
                                                     shape=(1, *shape),
                                                     initializer=I.ConstantInitializer(0.0))
            self._var = nn.parameter.get_parameter_or_create(name='var',
                                                             shape=(1, *shape),
                                                             initializer=I.ConstantInitializer(1.0))
            self._count = nn.parameter.get_parameter_or_create(name='count',
                                                               shape=(1, 1),
                                                               initializer=I.ConstantInitializer(1e-4))

    def process(self, x):
        std = (self._var + self._epsilon) ** 0.5
        normalized = (x - self._mean) / std
        if self._value_clip is not None:
            normalized = F.clip_by_value(
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
