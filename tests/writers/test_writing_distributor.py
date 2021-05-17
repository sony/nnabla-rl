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

import numpy as np

from nnabla_rl.writers.writing_distributor import WritingDistributor


class TestWritingDistributor():
    def test_write_scalar(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        distributor.write_scalar(1, {'test': 100})

        for writer in writers:
            writer.write_scalar.assert_called_once()

    def test_write_histogram(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        distributor.write_histogram(1, {'test': [100, 100]})

        for writer in writers:
            writer.write_histogram.assert_called_once()

    def test_write_image(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        image = np.empty(shape=(3, 10, 10))
        distributor.write_image(1, {'test': image})

        for writer in writers:
            writer.write_image.assert_called_once()

    def _new_mock_writer(self):
        return mock.MagicMock()
