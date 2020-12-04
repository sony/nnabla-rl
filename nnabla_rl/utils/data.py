from typing import List, TypeVar, Union, Iterable

import numpy as np

T = TypeVar('T')


def marshall_experiences(experiences):
    unzipped_experiences = unzip(experiences)
    return tuple(np.asarray(data) for data in unzipped_experiences)


def unzip(zipped_data):
    return list(zip(*zipped_data))


def is_array_like(x):
    return hasattr(x, "__len__")


def convert_to_list_if_not_iterable(value: Union[Iterable[T], T]) -> List[T]:
    if is_array_like(value):
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
