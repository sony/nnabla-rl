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

from nnabla.monitor import Monitor, MonitorSeries
from nnabla_rl.writer import Writer


class MonitorWriter(Writer):
    def __init__(self, outdir, file_prefix):
        super(MonitorWriter, self).__init__()
        self._file_prefix = file_prefix
        self._monitor = Monitor(str(outdir))
        self._monitors = {}

    def write_scalar(self, iteration_num, scalar):
        prefix = self._file_prefix + '_scalar_'
        for name, value in scalar.items():
            monitor = self._create_or_get_monitor_series(prefix + name)
            monitor.add(iteration_num, value)

    def write_histogram(self, iteration_num, histogram):
        pass

    def write_image(self, iteration_num, image):
        pass

    def _create_or_get_monitor_series(self, name):
        if name in self._monitors:
            return self._monitors[name]
        monitor = MonitorSeries(name, self._monitor, verbose=False)
        self._monitors[name] = monitor
        return monitor
