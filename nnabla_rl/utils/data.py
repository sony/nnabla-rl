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

from typing import Iterable, List, TypeVar, Union

import numpy as np

T = TypeVar('T')


def add_axis_if_single_dim(data):
    if data.ndim == 1:
        return np.expand_dims(data, axis=-1)
    else:
        return data


def marshal_experiences(experiences):
    unzipped_experiences = unzip(experiences)
    return tuple(add_axis_if_single_dim(np.asarray(data)) for data in unzipped_experiences)


def unzip(zipped_data):
    return list(zip(*zipped_data))


def is_array_like(x):
    return hasattr(x, "__len__")


def convert_to_list_if_not_list(value: Union[Iterable[T], T]) -> List[T]:
    if isinstance(value, Iterable):
        return list(value)
    else:
        return [value]


class RingBuffer(object):
    def __init__(self, maxlen):
        # Do NOT replace this list with collections.deque.
        # deque is too slow when randomly accessed to sample data for creating batch
        self._buffer = [None for _ in range(maxlen)]
        self._maxlen = maxlen
        self._head = 0
        self._length = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise KeyError
        return self._buffer[(self._head + index) % self._maxlen]

    def append(self, data):
        self.append_with_removed_item_check(data)

    def append_with_removed_item_check(self, data):
        if self._length < self._maxlen:
            self._length += 1
        elif self._length == self._maxlen:
            self._head = (self._head + 1) % self._maxlen
        else:
            raise IndexError
        index = (self._head + self._length - 1) % self._maxlen
        removed = self._buffer[index]
        self._buffer[index] = data
        return removed
