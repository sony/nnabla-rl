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


class BufferIterator(object):
    '''
    Simple iterator for iterating through the replay buffer.
    Replay buffer must support indexing.
    '''

    def __init__(self, buffer, batch_size, shuffle=True, repeat=True):
        super(BufferIterator, self).__init__()
        self._replay_buffer = buffer
        self._new_epoch = False
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._repeat = repeat
        self._indices = None

        self._index = 0
        self.reset()

    def __iter__(self):
        return self

    def next(self):
        if self.is_new_epoch():
            self._new_epoch = False
            raise StopIteration
        indices = \
            self._indices[self._index:self._index + self._batch_size]
        if (len(indices) < self._batch_size):
            if self._repeat:
                rest = self._batch_size - len(indices)
                self.reset()
                indices = np.append(
                    indices, self._indices[self._index:self._index + rest])
                self._index += rest
            else:
                self._index = len(self._replay_buffer)
            self._new_epoch = True
        else:
            self._index += self._batch_size
            self._new_epoch = (len(self._replay_buffer) <= self._index)
        return self._replay_buffer.sample_indices(indices)

    __next__ = next

    def is_new_epoch(self):
        return self._new_epoch

    def reset(self):
        self._indices = np.arange(len(self._replay_buffer))
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._index = 0
