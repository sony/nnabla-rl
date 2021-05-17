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

from nnabla_rl.hook import Hook
from nnabla_rl.utils.serializers import save_snapshot


class SaveSnapshotHook(Hook):
    '''
    Hook to save the training snapshot of current algorithm.

    Args:
        timing (int): Saving interval. Defaults to 1000 iteration.
    '''

    def __init__(self, outdir, timing=1000):
        super(SaveSnapshotHook, self).__init__(timing=timing)
        self._outdir = outdir

    def on_hook_called(self, algorithm):
        save_snapshot(self._outdir, algorithm)
