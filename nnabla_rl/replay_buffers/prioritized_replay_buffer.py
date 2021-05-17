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

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer


@dataclass
class Node:
    parent: int = -1
    left: int = 1
    right: int = 2
    value: float = 0.0


class SumTree(object):
    def __init__(self, capacity, init_max_p=1.0):
        self._capacity = capacity

        self._data = np.zeros(capacity, dtype=object)
        self._tree = [self._make_init_node(i) for i in range(2*capacity-1)]
        self._index = 0
        self._data_num = 0
        self._min_p = math.inf
        self._max_p = init_max_p
        self._latest_indices = None

    def _make_init_node(self, index):
        parent = (index - 1) // 2
        left = 2 * index + 1 if index < self._capacity - 1 else -1
        right = left + 1 if index < self._capacity - 1 else -1
        value = 0.
        return Node(parent, left, right, value)

    def append(self, data):
        self._data[self._index] = data
        self.update(self._index, self._max_p)

        self._index = (self._index + 1) % self._capacity
        if self._data_num < self._capacity:
            self._data_num += 1

    def update(self, index, p):
        tree_index = index + self._capacity - 1
        change_p = p - self._tree[tree_index].value
        self._tree[tree_index].value = float(p)
        self._update_parent(tree_index, change_p)

        self._min_p = min(self._min_p, p)
        self._max_p = max(self._max_p, p)

    def _update_parent(self, index, change_p):
        if index > 0:
            parent = self._tree[index].parent
            self._tree[parent].value += change_p
            self._update_parent(parent, change_p)

    def sample(self, num_samples=1, beta=0.6):
        random_values = np.random.uniform(0.0, self.total, size=num_samples)
        indices = [self._get_data_index_from_query(v) for v in random_values]
        return self.sample_indices(indices, beta)

    def sample_indices(self, indices, beta=0.6):
        if self._latest_indices is not None:
            raise RuntimeError('Trying to sample data from buffer without updating priority. '
                               'Check that the algorithm supports prioritized replay buffer.')
        data = [self._data[i] for i in indices]
        priorities = np.array([self._get_priority(i)
                               for i in indices])[:, np.newaxis]
        weights = self._weights_from_priorities(priorities, beta)
        self._latest_indices = indices
        return data, weights

    def _get_data_index_from_query(self, query):
        node = self._tree[0]
        while node.left >= 0:
            left_value = self._tree[node.left].value
            if query < left_value:
                index = node.left
            else:
                index = node.right
                query -= left_value
            node = self._tree[index]
        data_index = index - (self._capacity - 1)
        return data_index

    def _get_priority(self, index):
        tree_index = index + self._capacity - 1
        return self._tree[tree_index].value

    def _weights_from_priorities(self, priorities, beta):
        weights = (priorities / self._min_p) ** (-beta)
        return weights

    def update_latest_priorities(self, priorities):
        for index, priority in zip(self._latest_indices, priorities):
            self.update(index, priority)
        self._latest_indices = None

    def __len__(self):
        return self._data_num

    def __getitem__(self, index):
        return self._data[index]

    @property
    def total(self):
        return self._tree[0].value


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self, capacity,
        alpha=0.6, beta=0.4, betasteps=10000, epsilon=1e-8
    ):
        # No need to call super class contructor
        self._capacity_check(capacity)
        self._capacity = capacity
        self._buffer = SumTree(capacity)
        self._alpha = alpha
        self._beta = beta
        self._beta_diff = (1.0 - beta) / betasteps
        self._epsilon = epsilon

    def _capacity_check(self, capacity):
        if capacity is None or capacity <= 0:
            error_msg = 'buffer size must be greater than 0'
            raise ValueError(error_msg)

    def append(self, experience):
        self._buffer.append(experience)

    def sample(self, num_samples=1):
        buffer_length = len(self)
        if num_samples > buffer_length:
            error_msg = 'num_samples: {} is greater than the size of buffer: {}'.format(
                num_samples, buffer_length)
            raise ValueError(error_msg)
        experiences, weights = self._buffer.sample(num_samples, self._beta)
        info = dict(weights=weights)
        self._beta = min(self._beta + self._beta_diff, 1.0)
        return experiences, info

    def sample_indices(self, indices):
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        experiences, weights = self._buffer.sample_indices(indices, self._beta)
        info = dict(weights=weights)
        self._beta = min(self._beta + self._beta_diff, 1.0)
        return experiences, info

    def update_priorities(self, errors):
        priorities = ((errors + self._epsilon) ** self._alpha).flatten()
        self._buffer.update_latest_priorities(priorities)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]
