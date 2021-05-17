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

import pprint

from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class IterationStateHook(Hook):
    '''
    Hook which retrieves the iteration state to print/save the training status through writer.

    Args:
        timing (int): Retriving interval. Defaults to 1000 iteration.
        writer (nnabla_rl.writer.Writer, optional): Writer instance to save/print the iteration states.
            Defaults to None.
    '''

    def __init__(self, writer=None, timing=1000):
        self._timing = timing
        self._writer = writer

    def on_hook_called(self, algorithm):
        logger.info('Iteration state at iteration {}'.format(
            algorithm.iteration_num))

        latest_iteration_state = algorithm.latest_iteration_state

        if 'scalar' in latest_iteration_state:
            logger.info(pprint.pformat(latest_iteration_state['scalar']))

        if self._writer is not None:
            for key, value in latest_iteration_state.items():
                if key == 'scalar':
                    self._writer.write_scalar(algorithm.iteration_num, value)
                if key == 'histogram':
                    self._writer.write_histogram(algorithm.iteration_num, value)
                if key == 'image':
                    self._writer.write_image(algorithm.iteration_num, value)
