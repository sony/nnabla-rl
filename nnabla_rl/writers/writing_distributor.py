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

from nnabla_rl.writer import Writer


class WritingDistributor(Writer):
    def __init__(self, writers):
        super(WritingDistributor, self).__init__()
        self._writers = writers

    def write_scalar(self, iteration_num, scalar):
        for writer in self._writers:
            writer.write_scalar(iteration_num, scalar)

    def write_histogram(self, iteration_num, histogram):
        for writer in self._writers:
            writer.write_histogram(iteration_num, histogram)

    def write_image(self, iteration_num, image):
        for writer in self._writers:
            writer.write_image(iteration_num, image)

    def _write_file_header(self, filepath, keys):
        with open(filepath, 'w+') as f:
            np.savetxt(f, [list(keys)], fmt='%s', delimiter='\t')
