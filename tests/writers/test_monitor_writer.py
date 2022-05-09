# Copyright 2022 Sony Group Corporation.
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

import os
import tempfile

import numpy as np

from nnabla_rl.writers.monitor_writer import MonitorWriter


class TestMonitorWriter():
    def test_write_scalar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_returns = np.arange(5)
            test_results = {}
            test_results['mean'] = np.mean(test_returns)
            test_results['std_dev'] = np.std(test_returns)
            test_results['min'] = np.min(test_returns)
            test_results['max'] = np.max(test_returns)
            test_results['median'] = np.median(test_returns)

            writer = MonitorWriter(
                outdir=tmpdir, file_prefix='evaluation_results')
            writer.write_scalar(1, test_results)

            for name in test_results.keys():
                file_name = f'evaluation_results_scalar_{name}.series.txt'
                file_path = os.path.join(tmpdir, file_name)
                this_file_dir = os.path.dirname(__file__)
                test_file_dir = this_file_dir.replace('tests', 'test_resources')
                test_file_path = os.path.join(test_file_dir, file_name)
                self._check_same_txt_file(file_path, test_file_path)

    def _check_same_txt_file(self, file_path1, file_path2):
        # check each line
        with open(file_path1, mode='rt') as data_1, \
                open(file_path2, mode='rt') as data_2:
            for d_1, d_2 in zip(data_1, data_2):
                assert d_1 == d_2


if __name__ == "__main__":
    import pytest
    pytest.main()
