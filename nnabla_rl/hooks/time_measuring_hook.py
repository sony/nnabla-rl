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

import time

from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class TimeMeasuringHook(Hook):
    '''
    Hook to measure and print the actual time spent to run the iteration(s).

    Args:
        timing (int): Measuring interval. Defaults to 1 iteration.
    '''

    def __init__(self, timing=1):
        super(TimeMeasuringHook, self).__init__(timing=timing)
        self._prev_time = time.time()

    def on_hook_called(self, algorithm):
        current_time = time.time()

        logger.info("time spent since previous hook: {} seconds".format(
            current_time - self._prev_time))

        self._prev_time = current_time
