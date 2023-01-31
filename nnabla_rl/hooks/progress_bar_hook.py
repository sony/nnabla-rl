# Copyright 2022,2023 Sony Group Corporation.
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

from tqdm import tqdm

from nnabla_rl.algorithm import Algorithm
from nnabla_rl.hook import Hook


class ProgressBarHook(Hook):
    """Hook to show progress bar.

    Args:
        timing (int): Updating interval. Defaults to 1 iteration.
    """

    def __init__(self, timing: int = 1):
        super(ProgressBarHook, self).__init__(timing=timing)

    def on_hook_called(self, algorithm: Algorithm):
        if algorithm.iteration_num != 0:
            self._progress_bar.update(self._timing)

    def setup(self, algorithm: Algorithm, total_iterations: int):
        self._progress_bar = tqdm(total=total_iterations, position=-1)

    def teardown(self, algorithm: Algorithm, total_iterations: int):
        self._progress_bar.close()
