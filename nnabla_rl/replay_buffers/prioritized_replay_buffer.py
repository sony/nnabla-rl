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

import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np

import nnabla_rl as rl
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience
from nnabla_rl.utils.data import RingBuffer


@dataclass
class Node:
    parent: int = -1
    left: int = 1
    right: int = 2
    value: float = 0.0


class SumTree(object):
    def __init__(self, capacity, init_max_p=1.0):
        self._capacity = capacity

        # Ring buffer indices := relative indices (the oldest index is 0 and the newest is capacity - 1)
        # Actual data indices := absolute indices
        self._data = RingBuffer(maxlen=capacity)
        self._tree = [self._make_init_node(i) for i in range(2*capacity-1)]

        # tail index of RingBuffer
        self._tail_index = 0
        self._min_p = math.inf
        self._max_p = init_max_p

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    @property
    def total(self):
        return self._tree[0].value

    def update_priorities(self, relative_indices, priorities):
        for relative_index, priority in zip(relative_indices, priorities):
            absolute_index = self._relative_to_absolute_index(relative_index)
            self._update(absolute_index, priority)

    def get_index_from_query(self, query):
        node = self._tree[0]
        while node.left >= 0:
            left_value = self._tree[node.left].value
            if query < left_value:
                tree_index = node.left
            else:
                tree_index = node.right
                query -= left_value
            node = self._tree[tree_index]
        absolute_index = self._tree_to_absolute_index(tree_index)
        return self._absolute_to_relative_index(absolute_index)

    def append(self, data):
        self._data.append(data)
        self._update(self._tail_index, self._max_p)

        self._tail_index = (self._tail_index + 1) % self._capacity

    def append_with_removed_item_check(self, data):
        removed = self._data.append_with_removed_item_check(data)
        self._update(self._tail_index, self._max_p)
        self._tail_index = (self._tail_index + 1) % self._capacity
        return removed

    def get_priority(self, relative_index):
        absolute_index = self._relative_to_absolute_index(relative_index)
        tree_index = self._absolute_to_tree_index(absolute_index)
        return self._tree[tree_index].value

    def weights_from_priorities(self, priorities, beta):
        weights = (priorities / self._min_p) ** (-beta)
        return weights

    def _make_init_node(self, index):
        parent = (index - 1) // 2
        left = 2 * index + 1 if index < self._capacity - 1 else -1
        right = left + 1 if index < self._capacity - 1 else -1
        value = 0.
        return Node(parent, left, right, value)

    def _update(self, absolute_index, p):
        tree_index = self._absolute_to_tree_index(absolute_index)
        change_p = p - self._tree[tree_index].value
        self._tree[tree_index].value = float(p)
        self._update_parent(tree_index, change_p)

        self._min_p = min(self._min_p, p)
        self._max_p = max(self._max_p, p)

    def _update_parent(self, tree_index, change_p):
        if tree_index > 0:
            parent = self._tree[tree_index].parent
            self._tree[parent].value += change_p
            self._update_parent(parent, change_p)

    def _relative_to_absolute_index(self, relative_index):
        return (relative_index + self._data._head) % self._capacity

    def _absolute_to_relative_index(self, absolute_index):
        return (absolute_index - self._data._head) % self._capacity

    def _absolute_to_tree_index(self, absolute_index):
        return absolute_index + self._capacity - 1

    def _tree_to_absolute_index(self, tree_index):
        return tree_index - (self._capacity - 1)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, betasteps=10000, epsilon=1e-8):
        # No need to call super class contructor
        self._capacity_check(capacity)
        self._capacity = capacity
        self._buffer = SumTree(capacity)
        self._alpha = alpha
        self._beta = beta
        self._beta_diff = (1.0 - beta) / betasteps
        self._epsilon = epsilon

        self._last_sampled_indices = None  # last absolute indices of experiences sampled from buffer

    def _capacity_check(self, capacity):
        if capacity is None or capacity <= 0:
            error_msg = 'buffer size must be greater than 0'
            raise ValueError(error_msg)

    def append(self, experience):
        if self._last_sampled_indices is not None:
            raise RuntimeError('Trying to append data to buffer without updating priority. '
                               'Check that the algorithm supports prioritized replay buffer.')
        self._buffer.append(experience)

    def sample(self, num_samples=1, num_steps=1):
        buffer_length = len(self)
        if num_samples > buffer_length:
            error_msg = 'num_samples: {} is greater than the size of buffer: {}'.format(
                num_samples, buffer_length)
            raise ValueError(error_msg)
        if buffer_length - num_steps < 0:
            raise RuntimeError(f'Insufficient buffer length. buffer: {buffer_length} < steps: {num_steps}')
        indices = []
        while len(indices) < num_samples:
            random_value = rl.random.drng.uniform(0.0, self._buffer.total)
            index = self._buffer.get_index_from_query(random_value)
            if index < buffer_length - num_steps + 1:
                indices.append(index)
        return self.sample_indices(indices, num_steps)

    def sample_indices(self, indices, num_steps=1):
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        if self._last_sampled_indices is not None:
            raise RuntimeError('Trying to sample data from buffer without updating priority. '
                               'Check that the algorithm supports prioritized replay buffer.')
        experiences: Union[Sequence[Experience], Tuple[Sequence[Experience], ...]]
        if num_steps == 1:
            experiences = [self.__getitem__(index) for index in indices]
        else:
            experiences = tuple([self.__getitem__(index+i) for index in indices] for i in range(num_steps))

        priorities = np.asarray([self._buffer.get_priority(i) for i in indices])[:, np.newaxis]
        weights = self._buffer.weights_from_priorities(priorities, self._beta)

        self._last_sampled_indices = indices

        info = dict(weights=weights)
        self._beta = min(self._beta + self._beta_diff, 1.0)
        return experiences, info

    def update_priorities(self, errors):
        priorities = ((errors + self._epsilon) ** self._alpha).flatten()
        self._buffer.update_priorities(self._last_sampled_indices, priorities)
        self._last_sampled_indices = None

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]
