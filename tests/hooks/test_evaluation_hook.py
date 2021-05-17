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

from unittest import mock

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.hooks import EvaluationHook


class TestEvaluationHook():
    def test_call(self):
        dummy_env = E.DummyContinuous()

        dummy_algorithm = A.Dummy(dummy_env)

        mock_evaluator = mock.MagicMock()
        mock_evaluator.return_value = [0]

        mock_writer = mock.MagicMock()

        hook = EvaluationHook(dummy_env,
                              evaluator=mock_evaluator,
                              writer=mock_writer)

        hook(dummy_algorithm)

        inputs = (dummy_algorithm, dummy_env)

        mock_evaluator.assert_called_once_with(*inputs)
        hook._writer.write_scalar.assert_called_once()
