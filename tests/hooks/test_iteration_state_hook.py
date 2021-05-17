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

from nnabla_rl.hooks import IterationStateHook
from nnabla_rl.writer import Writer


class TestIterationStateHook():
    def test_call(self):
        dummy_algorithm = mock.MagicMock()

        test_latest_iteration_state = {}
        test_latest_iteration_state['scalar'] = {}
        test_latest_iteration_state['histogram'] = {}
        test_latest_iteration_state['image'] = {}

        dummy_algorithm.iteration_num = 1
        dummy_algorithm.latest_iteration_state = test_latest_iteration_state

        writer = Writer()
        writer.write_scalar = mock.MagicMock()
        writer.write_histogram = mock.MagicMock()
        writer.write_image = mock.MagicMock()

        hook = IterationStateHook(writer=writer, timing=1)

        hook(dummy_algorithm)

        writer.write_scalar.assert_called_once_with(dummy_algorithm.iteration_num,
                                                    test_latest_iteration_state['scalar'])
        writer.write_histogram.assert_called_once_with(dummy_algorithm.iteration_num,
                                                       test_latest_iteration_state['histogram'])
        writer.write_image.assert_called_once_with(dummy_algorithm.iteration_num,
                                                   test_latest_iteration_state['image'])
