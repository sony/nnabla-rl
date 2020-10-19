import pytest

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F

from nnabla_rl.utils.debugging import count_parameter_number


class TestCountParameterNumber():
    @pytest.mark.parametrize("batch_size, state_size, output_size", [
        (5, 3, 2)])
    def test_affine_count(self, batch_size, state_size, output_size):
        nn.clear_parameters()
        dummy_input = nn.Variable((batch_size, state_size))

        with nn.parameter_scope("dummy_affine"):
            _ = F.relu(PF.affine(dummy_input, output_size))

        parameter_number = count_parameter_number(nn.get_parameters())

        assert parameter_number == state_size*output_size + output_size
