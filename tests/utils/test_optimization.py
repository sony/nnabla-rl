import pytest

import numpy as np

from nnabla_rl.utils.optimization import conjugate_gradient


class TestOptimization():
    def test_conjugate_gradient(self):
        x_dim = 2
        A = np.random.uniform(-3, 3, (x_dim, x_dim))
        symmetric_positive_A = np.dot(A, A.T)
        def compute_Ax(x): return np.dot(symmetric_positive_A, x)
        b = np.random.uniform(-3, 3, (x_dim, ))

        optimized_x = conjugate_gradient(
            compute_Ax, b, max_iterations=None)

        expected_x = np.dot(np.linalg.inv(symmetric_positive_A), b)

        assert expected_x == pytest.approx(optimized_x)
