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

import pytest

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
from nnabla_rl.utils.debugging import count_parameter_number


class TestCountParameterNumber():
    @pytest.mark.parametrize("batch_size, state_size, output_size", [
        (5, 3, 2)])
    def test_affine_count(self, batch_size, state_size, output_size):
        nn.clear_parameters()
        dummy_input = nn.Variable((batch_size, state_size))

        with nn.parameter_scope("dummy_affine"):
            _ = NF.relu(NPF.affine(dummy_input, output_size))

        parameter_number = count_parameter_number(nn.get_parameters())

        assert parameter_number == state_size*output_size + output_size
