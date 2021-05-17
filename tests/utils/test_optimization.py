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
import pytest

from nnabla_rl.utils.optimization import conjugate_gradient


class TestOptimization():
    def test_conjugate_gradient(self):
        x_dim = 2
        A = np.random.uniform(-3, 3, (x_dim, x_dim))
        symmetric_positive_A = np.dot(A, A.T)
        def compute_Ax(x): return np.dot(symmetric_positive_A, x)
        b = np.random.uniform(-3, 3, (x_dim, ))

        optimized_x = conjugate_gradient(
            compute_Ax, b, max_iterations=1000)

        expected_x = np.dot(np.linalg.inv(symmetric_positive_A), b)

        assert expected_x == pytest.approx(optimized_x)
