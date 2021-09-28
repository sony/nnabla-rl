
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

from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class EpochNumHook(Hook):
    '''
    Hook to print the epoch number periodically.
    This hook just prints the epoch number every iteration_per_epoch number of iteration.

    Args:
        iteration_per_epoch (int): Printing epoch interval. Defaults to 1 iteration.
    '''

    def __init__(self, iteration_per_epoch=1):
        super(EpochNumHook, self).__init__(timing=iteration_per_epoch)
        self._iteration_per_epoch = iteration_per_epoch

    def on_hook_called(self, algorithm):
        logger.info(f"current epoch: {algorithm.iteration_num // self._iteration_per_epoch}")
