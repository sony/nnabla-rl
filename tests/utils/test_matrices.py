import pytest

import numpy as np

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.initializer as I

from nnabla_rl.utils.matrices import compute_hessian
from nnabla.parameter import get_parameter_or_create


class TestComputeHessian():
    def setup_method(self, method):
        nn.clear_parameters()

    def test_compute_hessian(self):
        x = get_parameter_or_create("x", shape=(1, ))
        y = get_parameter_or_create("y", shape=(1, ))
        loss = x**3 + 2.*x*y + y**2 - x

        x.d = 2.
        y.d = 3.
        actual = compute_hessian(loss, nn.get_parameters().values())

        assert np.array([[12., 2.], [2., 2.]]) == pytest.approx(actual)

    def test_compute_network_parameters(self):
        state = nn.Variable((1, 2))
        output = PF.affine(state, 1, w_init=I.ConstantInitializer(
            value=1.), b_init=I.ConstantInitializer(value=1.))

        loss = F.sum(output**2)
        state_array = np.array([[1.0, 0.5]])
        state.d = state_array

        actual = compute_hessian(loss, nn.get_parameters().values())

        expected = np.array(
            [[2*state_array[0, 0]**2,
              2*state_array[0, 0]*state_array[0, 1],
              2*state_array[0, 0]],
             [2*state_array[0, 0]*state_array[0, 1],
              2*state_array[0, 1]**2,
              2*state_array[0, 1]],
             [2*state_array[0, 0],
              2*state_array[0, 1],
              2.]]
        )

        assert expected == pytest.approx(actual)
