# Copyright 2024 Sony Group Corporation.
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

from typing import Any, Sequence

import nnabla_rl as rl
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience
from nnabla_rl.utils.data import DataHolder


class RandomRemovalBuffer(DataHolder[Any]):
    def __init__(self, maxlen: int):
        self._buffer = [None for _ in range(maxlen)]
        self._maxlen = maxlen
        self._length = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise KeyError
        return self._buffer[index % self._maxlen]

    def append(self, data):
        assert float(data[0][2]) == 1.0
        self.append_with_removed_item_check(data)

    def append_with_removed_item_check(self, data):
        if self._length < self._maxlen:
            index = self._length
            self._length += 1
        elif self._length == self._maxlen:
            index = rl.random.drng.choice(self._maxlen)
        else:
            raise IndexError
        removed = self._buffer[index]
        self._buffer[index] = data
        return removed


class RandomRemovalReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 100000):
        assert capacity is None or capacity >= 0
        self._capacity = capacity
        self._buffer = RandomRemovalBuffer(maxlen=capacity)

    def append(self, experience: Experience):
        state = experience[0]
        assert len(state) == 3 or len(state) == 7
        # NOTE: if state is valid and add it
        if float(state[2]) == 1.0:
            self._buffer.append(experience)

    def append_all(self, experiences: Sequence[Experience]):
        for experience in experiences:
            assert len(experience) == 12
            self.append(experience)
