# Copyright 2021 Sony Corporation.
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

import random

from nnabla_rl.replay_buffer import ReplayBuffer


class ReplacementSamplingReplayBuffer(ReplayBuffer):
    '''ReplacementSamplingReplayBuffer.
    From all experiences in the buffer, this buffer samples the experiences with replacement.
    Therefore, sampled batch may contain duplicate (=same experience) entries.
    Unlike the default ReplayBuffer, you can sample larger number of data than total size of the buffer.
    '''

    def __init__(self, capacity=None):
        super(ReplacementSamplingReplayBuffer, self).__init__(capacity)

    def sample(self, num_samples=1):
        indices = self._random_indices(num_samples=num_samples)
        return self.sample_indices(indices)

    def _random_indices(self, num_samples):
        buffer_length = len(self)
        return random.choices(range(buffer_length), k=num_samples)
