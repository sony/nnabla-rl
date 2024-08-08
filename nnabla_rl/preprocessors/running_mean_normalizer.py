# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from typing import Optional, Tuple, Union

import numpy as np

import nnabla as nn
import nnabla.initializer as NI
from nnabla_rl.functions import compute_std, normalize
from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.typing import Shape


class RunningMeanNormalizer(Preprocessor, Model):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _shape: Shape
    """Running mean normalizer. This normalizer computes a running mean and \
    variance from the data and use them to process the given varible.

    Args:
        scope_name (str): scope name of running mean normalizer's parameters
        shape (Shape): shape of the variable which is normalized.
        epsilon (float): value to improve numerical stability for computing the standard deviation. Defaults to 1e-8.
        value_clip (Optional[Tuple[float, float]]): value clip. This clipping is applied after the normalization.
        mode_for_floating_point_error (str): mode for avoiding a floating point error \
            when computing the standard deviation from the variance to normalize the data. \
            Must be one of:

            - `add`: Use the square root of the sum of var and epsilon as the standard deviation.
            - `max`: Use the epsilon if the square root of var is smaller than epsilon, \
                otherwise it returns the square root of var as the standard deviation.

            Defaults to add.
        mean_initializer (Union[NI.BaseInitializer, np.ndarray]): initializer for normalizer's mean. \
            The computation of a running mean is started from this value. Defaults to NI.ConstantInitializer(0.0).
        var_initializer (Union[NI.BaseInitializer, np.ndarray]): initializer for normalizer's variance. \
            The computation of a running variance is started from this value. Defaults to NI.ConstantInitializer(1.0).
    """

    def __init__(
        self,
        scope_name: str,
        shape: Shape,
        epsilon: float = 1e-8,
        value_clip: Optional[Tuple[float, float]] = None,
        mode_for_floating_point_error: str = "add",
        mean_initializer: Union[NI.BaseInitializer, np.ndarray] = NI.ConstantInitializer(0.0),
        var_initializer: Union[NI.BaseInitializer, np.ndarray] = NI.ConstantInitializer(1.0),
    ):
        super(RunningMeanNormalizer, self).__init__(scope_name)

        if value_clip is not None and value_clip[0] > value_clip[1]:
            raise ValueError(f"Unexpected clipping value range: {value_clip[0]} > {value_clip[1]}")
        self._value_clip = value_clip

        if isinstance(shape, int):
            self._shape = (shape,)
        elif isinstance(shape, tuple):
            self._shape = shape
        else:
            raise ValueError

        self._epsilon = epsilon
        self._mode_for_floating_point_error = mode_for_floating_point_error

        if isinstance(mean_initializer, np.ndarray):
            assert mean_initializer.shape == shape
            mean_initializer = mean_initializer[np.newaxis, :]
        self._mean_initializer = mean_initializer

        if isinstance(var_initializer, np.ndarray):
            assert var_initializer.shape == shape
            var_initializer = var_initializer[np.newaxis, :]
        self._var_initializer = var_initializer

    def process(self, x):
        assert 0 < self._count.d
        std = compute_std(self._var, self._epsilon, self._mode_for_floating_point_error)
        normalized = normalize(x, self._mean, std, self._value_clip)
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
            return nn.parameter.get_parameter_or_create(
                name="mean", shape=(1, *self._shape), initializer=self._mean_initializer
            )

    @property
    def _var(self):
        with nn.parameter_scope(self.scope_name):
            return nn.parameter.get_parameter_or_create(
                name="var", shape=(1, *self._shape), initializer=self._var_initializer
            )

    @property
    def _count(self):
        with nn.parameter_scope(self.scope_name):
            return nn.parameter.get_parameter_or_create(
                name="count", shape=(1, 1), initializer=NI.ConstantInitializer(1e-4)
            )
