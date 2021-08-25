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
import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

import nnabla_rl as rl
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience
from nnabla_rl.utils.data import DataHolder, RingBuffer

T = TypeVar('T')


# NOTE: index naming convention used in this module
# relative index: 0: oldest item's index. capacity - 1: newest item's index.
# absolute index: actual data index in list. 0: list's head. capacity - 1: list's tail.
# tree index: 0: root of the tree. 2 * capacity - 1: right most leaf of the tree.
# heap index: 0: head of the heap. If max heap, maximum value is saved in this index. capacity - 1: tail of the heap.

@dataclass
class Node(Generic[T]):
    value: T
    parent: int = -1
    left: int = 1
    right: int = 2


class BinaryTree(Generic[T]):
    """ Common Binary Tree Class
    SumTree and MinTree is derived from this class.
    Args:
        capacity (int): the maximum number of saved data.
        init_node_value (T): the initial value of node.
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _tree: List[Node[T]]
    _tail_index: int

    def __init__(self, capacity: int, init_node_value: T):
        self._capacity = capacity
        self._init_node_value = init_node_value
        self._tail_index = 0
        self._length = 0
        self._tree = [self._make_init_node(i) for i in range(2*capacity-1)]

    def __len__(self):
        return self._length

    def __getitem__(self, tree_index: int):
        return self._tree[tree_index].value

    def append(self, value: T):
        self.update(self._tail_index, value)
        self._tail_index = (self._tail_index + 1) % self._capacity
        if self._length < self._capacity:
            self._length += 1

    def update(self, absolute_index: int, value: T):
        tree_index = self.absolute_to_tree_index(absolute_index)
        self._tree[tree_index].value = value
        self._update_parent(tree_index)

    @abstractmethod
    def _update_parent(self, tree_index: int):
        raise NotImplementedError

    def tree_to_absolute_index(self, tree_index: int):
        return tree_index - (self._capacity - 1)

    def absolute_to_tree_index(self, absolute_index: int):
        return absolute_index + self._capacity - 1

    def _make_init_node(self, index: int):
        parent = (index - 1) // 2
        left = 2 * index + 1 if index < self._capacity - 1 else -1
        right = left + 1 if index < self._capacity - 1 else -1
        value = self._init_node_value
        return Node(value=value, parent=parent, left=left, right=right)


class MinTree(BinaryTree[float]):
    def __init__(self, capacity: int):
        super(MinTree, self).__init__(capacity, init_node_value=math.inf)

    def min(self):
        return self._tree[0].value

    def _update_parent(self, tree_index: int):
        if tree_index > 0:
            parent_index = self._tree[tree_index].parent
            left_index = self._tree[parent_index].left
            left_value = self._tree[left_index].value
            right_index = self._tree[parent_index].right
            right_value = self._tree[right_index].value
            self._tree[parent_index].value = min(left_value, right_value)
            self._update_parent(parent_index)


class SumTree(BinaryTree[float]):
    def __init__(self, capacity: int):
        super(SumTree, self).__init__(capacity, init_node_value=0.0)

    def get_absolute_index_from_query(self, query: float):
        """ Sample absolute index from query value
        """
        if query < 0 or query > self.sum():
            raise ValueError(f"You must use value between [0, {self.sum()}] as query")
        node = self._tree[0]
        while node.left >= 0:
            left_value = self._tree[node.left].value
            if query < left_value:
                tree_index = node.left
            else:
                tree_index = node.right
                query -= left_value
            node = self._tree[tree_index]
        return self.tree_to_absolute_index(tree_index)

    def sum(self):
        return self._tree[0].value

    def _update_parent(self, tree_index: int):
        if tree_index > 0:
            parent_index = self._tree[tree_index].parent
            left_index = self._tree[parent_index].left
            left_value = self._tree[left_index].value
            right_index = self._tree[parent_index].right
            right_value = self._tree[right_index].value
            self._tree[parent_index].value = left_value + right_value
            self._update_parent(parent_index)


class MaxHeap(object):
    def __init__(self, capacity):
        self._capacity = capacity
        self._heap = [None for _ in range(capacity)]
        self._heap_to_absolute_index_map = [None for _ in range(capacity)]
        self._absolute_to_heap_index_map = [None for _ in range(capacity)]

        self._tail_index = 0
        self._oldest_index = 0
        self._length = 0

    def __len__(self):
        return self._length

    def __getitem__(self, heap_index: int):
        return self._heap[heap_index]

    def append(self, value: float):
        if len(self) == self._capacity:
            # remove the oldest and replace with new data
            # Reset the priority of oldest_index data to maximum
            # We know that new data will be inserted there
            self.update(self._oldest_index, value)
            self._oldest_index = (self._oldest_index + 1) % self._capacity
        else:
            self._heappush(self._tail_index, value)
            if self._tail_index < self._capacity - 1:
                self._tail_index += 1
            self._length += 1

    def sort_data(self):
        # Decreasing order
        self._heap = sorted(self._heap, key=lambda item: -math.inf if item is None else item[1], reverse=True)

        # Reset index map
        for index, item in enumerate(self._heap):
            if item is not None:
                self._heap_to_absolute_index_map[index] = item[0]
                self._absolute_to_heap_index_map[item[0]] = index
            else:
                self._heap_to_absolute_index_map[index] = None

    def get_absolute_index_from_heap_index(self, heap_index: int):
        return self.heap_to_absolute_index(heap_index)

    def update(self, absolute_index: int, value: float):
        heap_index = self.absolute_to_heap_index(absolute_index)
        (absolute_index, _) = self._heap[heap_index]
        self._heap[heap_index] = (absolute_index, value)
        self._heapup(heap_index)
        self._heapdown(heap_index)

    def _parent_index(self, child_index):
        return (child_index - 1) // 2

    def _heappush(self, absolute_index, error):
        heap_index = self._tail_index
        self._heap_to_absolute_index_map[heap_index] = absolute_index
        self._absolute_to_heap_index_map[absolute_index] = heap_index
        self._heap[heap_index] = (absolute_index, error)
        self._heapup(heap_index)

    def _heapup(self, heap_index):
        if heap_index == 0:
            return
        heap_data = self._heap[heap_index]
        parent_index = self._parent_index(heap_index)
        parent_data = self._heap[parent_index]
        if parent_data[1] < heap_data[1]:
            self._swap_item(heap_index, parent_index)
            self._heapup(parent_index)

    def _heapdown(self, heap_index):
        heap_length = len(self)
        if heap_length <= heap_index:
            return
        heap_data = self._heap[heap_index]
        child_l_index = heap_index * 2 + 1
        child_r_index = heap_index * 2 + 2
        child_l_data = self._heap[child_l_index] if child_l_index < self._capacity else None
        child_r_data = self._heap[child_r_index] if child_r_index < self._capacity else None

        largest_data_index = heap_index
        if child_l_data is not None:
            if (child_l_index < heap_length) and (child_l_data[1] > heap_data[1]):
                largest_data_index = child_l_index
        if child_r_data is not None:
            if (child_r_index < heap_length) and (child_r_data[1] > self._heap[largest_data_index][1]):
                largest_data_index = child_r_index
        if largest_data_index != heap_index:
            self._swap_item(heap_index, largest_data_index)
            self._heapdown(largest_data_index)

    def _swap_item(self, heap_index1, heap_index2):
        heap_index1_data = self._heap[heap_index1]
        heap_index2_data = self._heap[heap_index2]
        self._heap[heap_index1], self._heap[heap_index2] = heap_index2_data, heap_index1_data
        self._heap_to_absolute_index_map[heap_index1] = heap_index2_data[0]
        self._absolute_to_heap_index_map[heap_index2_data[0]] = heap_index1
        self._heap_to_absolute_index_map[heap_index2] = heap_index1_data[0]
        self._absolute_to_heap_index_map[heap_index1_data[0]] = heap_index2

    def absolute_to_heap_index(self, absolute_index):
        return self._absolute_to_heap_index_map[absolute_index]

    def heap_to_absolute_index(self, heap_index):
        return self._heap_to_absolute_index_map[heap_index]


class PrioritizedDataHolder(DataHolder[Any]):
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._data = RingBuffer(maxlen=capacity)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, relative_index: int):
        return self._data[relative_index]

    def append(self, data):
        # ignore returned value
        self.append_with_removed_item_check(data)

    def append_with_removed_item_check(self, data):
        raise NotImplementedError

    def update_priority(self, relative_index: int, priority: int):
        raise NotImplementedError

    def get_priority(self, relative_index: int):
        raise NotImplementedError

    def _relative_to_absolute_index(self, relative_index):
        return (relative_index + self._data._head) % self._capacity

    def _absolute_to_relative_index(self, absolute_index):
        return (absolute_index - self._data._head) % self._capacity


class SumTreeDataHolder(PrioritizedDataHolder):
    def __init__(self, capacity, initial_max_priority, keep_min=True):
        super().__init__(capacity=capacity)
        self._sum_tree = SumTree(capacity=capacity)
        self._keep_min = keep_min
        if self._keep_min:
            self._min_tree = MinTree(capacity=capacity)
        self._max_priority = initial_max_priority

    def append_with_removed_item_check(self, data):
        removed = self._data.append_with_removed_item_check(data)
        self._sum_tree.append(self._max_priority)
        if self._keep_min:
            self._min_tree.append(self._max_priority)
        return removed

    def get_priority(self, relative_index: int):
        absolute_index = self._relative_to_absolute_index(relative_index)
        tree_index = self._sum_tree.absolute_to_tree_index(absolute_index)
        return self._sum_tree[tree_index]

    def sum_priority(self):
        return self._sum_tree.sum()

    def min_priority(self):
        return self._min_tree.min()

    def update_priority(self, relative_index: int, priority: float):
        absolute_index = self._relative_to_absolute_index(relative_index)
        self._sum_tree.update(absolute_index, priority)
        if self._keep_min:
            self._min_tree.update(absolute_index, priority)
        self._max_priority = max(self._max_priority, priority)

    def get_index_from_query(self, query: float):
        absolute_index = self._sum_tree.get_absolute_index_from_query(query)
        return self._absolute_to_relative_index(absolute_index)


class MaxHeapDataHolder(PrioritizedDataHolder):
    def __init__(self, capacity: int, alpha: float):
        super().__init__(capacity=capacity)
        self._max_heap = MaxHeap(capacity)
        self._alpha = alpha

    def append_with_removed_item_check(self, data):
        removed = self._data.append_with_removed_item_check(data)
        self._max_heap.append(math.inf)
        return removed

    def get_priority(self, relative_index: int):
        absolute_index = self._relative_to_absolute_index(relative_index)
        heap_index = self._max_heap.absolute_to_heap_index(absolute_index)
        rank = (heap_index + 1)
        return self._compute_priority(rank)

    def get_relative_index_from_heap_index(self, heap_index: int):
        absolute_index = self._max_heap.get_absolute_index_from_heap_index(heap_index)
        return self._absolute_to_relative_index(absolute_index)

    def update_priority(self, relative_index: int, priority: float):
        absolute_index = self._relative_to_absolute_index(relative_index)
        self._max_heap.update(absolute_index, priority)

    def sort_data(self):
        self._max_heap.sort_data()

    def _compute_priority(self, rank: int):
        priority = (1 / rank) ** self._alpha

        # We do not normalize priority here to reduce computation.
        # Normalization term will be compensated when dividing with maximum weight
        return priority


class _PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 capacity: int,
                 alpha: float,
                 beta: float,
                 betasteps: int,
                 error_clip: Optional[Tuple[float, float]]):
        # Do not call super class' constructor
        self._capacity_check(capacity)
        self._capacity = capacity

        self._alpha = alpha
        self._beta = beta
        self._beta_diff = (1.0 - beta) / betasteps

        self._error_clip = error_clip

        # last absolute indices of experiences sampled from buffer
        self._last_sampled_indices: Union[Sequence[int], None] = None

    def __getitem__(self, relative_index: int):
        # NOTE: relative index 0 means the oldest entry and len(self) - 1 the latest entry
        return self._buffer[relative_index]

    def __len__(self):
        return len(self._buffer)

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        raise NotImplementedError

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1):
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

        weights = self._get_weights(indices, self._alpha, self._beta)
        info = dict(weights=weights)

        self._beta = min(self._beta + self._beta_diff, 1.0)
        self._last_sampled_indices = indices
        return experiences, info

    def update_priorities(self, errors: np.ndarray):
        raise NotImplementedError

    def _preprocess_errors(self, errors: np.ndarray):
        if self._error_clip is not None:
            errors = np.clip(errors, self._error_clip[0], self._error_clip[1])
        return np.abs(errors)

    def _get_weights(self, indices: Sequence[int], alpha: float, beta: float):
        raise NotImplementedError

    def _capacity_check(self, capacity: int):
        if capacity is None or capacity <= 0:
            error_msg = 'buffer size must be greater than 0'
            raise ValueError(error_msg)


class ProportionalPrioritizedReplayBuffer(_PrioritizedReplayBuffer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _buffer: SumTreeDataHolder
    _epsilon: float

    def __init__(self, capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 betasteps: int = 10000,
                 error_clip: Optional[Tuple[float, float]] = (-1, 1),
                 epsilon: float = 1e-8,
                 init_max_error: float = 1.0,
                 normalization_method: str = "buffer_max"):
        super(ProportionalPrioritizedReplayBuffer, self).__init__(capacity, alpha, beta, betasteps, error_clip)
        assert normalization_method in ("batch_max", "buffer_max")
        self._normalization_method = normalization_method
        keep_min = (self._normalization_method == "buffer_max")
        self._buffer = SumTreeDataHolder(capacity=capacity, initial_max_priority=init_max_error, keep_min=keep_min)
        self._epsilon = epsilon

    def append(self, experience):
        self._buffer.append(experience)

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        buffer_length = len(self)
        if num_samples > buffer_length:
            error_msg = f'num_samples: {num_samples} is greater than the size of buffer: {buffer_length}'
            raise ValueError(error_msg)
        if buffer_length - num_steps < 0:
            raise RuntimeError(f'Insufficient buffer length. buffer: {buffer_length} < steps: {num_steps}')

        # In paper,
        # "To sample a minibatch of size k, the range [0, ptotal] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range"
        indices = []
        interval = self._buffer.sum_priority() / num_samples
        for i in range(num_samples):
            index = sys.maxsize
            while index >= buffer_length - num_steps + 1:
                random_value = rl.random.drng.uniform(interval * i, interval * (i + 1))
                index = self._buffer.get_index_from_query(random_value)
            indices.append(index)
        return self.sample_indices(indices, num_steps)

    def update_priorities(self, errors: np.ndarray):
        errors = self._preprocess_errors(errors)
        errors = ((errors + self._epsilon) ** self._alpha).flatten()
        indices = cast(Sequence[int], self._last_sampled_indices)
        for index, error in zip(indices, errors):
            self._buffer.update_priority(index, error)
        self._last_sampled_indices = None

    def _get_weights(self, indices: Sequence[int], alpha: float, beta: float):
        priorities = np.asarray([self._buffer.get_priority(i) for i in indices])[:, np.newaxis]
        if self._normalization_method == "batch_max":
            # Use min priority. This is same as max of weight.
            min_priority = priorities.min()
        elif self._normalization_method == "buffer_max":
            # Use min priority. This is same as max of weight.
            min_priority = self._buffer.min_priority()
        else:
            raise RuntimeError(f"Unknown normalization method {self._normalization_method}")
        return (priorities / min_priority) ** (-beta)


class RankBasedPrioritizedReplayBuffer(_PrioritizedReplayBuffer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _buffer: MaxHeapDataHolder
    _reset_segment_interval: int
    _sort_interval: int
    _boundaries: List[int]
    _prev_num_samples: int
    _prev_num_steps: int
    _appends_since_prev_start: int

    def __init__(self, capacity: int,
                 alpha: float = 0.7,
                 beta: float = 0.5,
                 betasteps: int = 10000,
                 error_clip: Optional[Tuple[float, float]] = (-1, 1),
                 reset_segment_interval: int = 1000,
                 sort_interval: int = 1000000):
        super(RankBasedPrioritizedReplayBuffer, self).__init__(capacity, alpha, beta, betasteps, error_clip)
        self._buffer = MaxHeapDataHolder(capacity, alpha)

        self._reset_segment_interval = reset_segment_interval
        self._sort_interval = sort_interval

        self._boundaries = []
        self._prev_num_samples = 0
        self._prev_num_steps = 0
        self._appends_since_prev_sort = 0
        self._ps_cumsum = np.cumsum(np.asarray([(1 / (i + 1)) ** alpha for i in range(capacity)]))

    def append(self, experience):
        self._buffer.append(experience)

        self._appends_since_prev_sort += 1
        if self._appends_since_prev_sort % self._sort_interval == 0:
            self._buffer.sort_data()
            self._appends_since_prev_sort = 0

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        buffer_length = len(self)
        if num_samples > buffer_length:
            error_msg = f'num_samples: {num_samples} is greater than the size of buffer: {buffer_length}'
            raise ValueError(error_msg)
        if buffer_length - num_steps < 0:
            raise RuntimeError(
                f'Insufficient buffer length. buffer: {buffer_length} < steps: {num_steps}')
        if (num_samples != self._prev_num_samples) or \
           (num_steps != self._prev_num_steps) or \
           (buffer_length % self._reset_segment_interval == 0 and buffer_length != self._capacity) or \
           (len(self._boundaries) == 0):
            self._boundaries = self._compute_segment_boundaries(N=buffer_length, k=num_samples)
            self._prev_num_samples = num_samples
            self._prev_num_steps = num_steps

        indices = []
        prev_boundary = 0
        for boundary in self._boundaries:
            heap_index = rl.random.drng.integers(low=prev_boundary, high=boundary)
            index = self._buffer.get_relative_index_from_heap_index(heap_index)
            prev_boundary = boundary
            if index < buffer_length - num_steps + 1:
                indices.append(index)
        while len(indices) < num_samples:
            # Enters here only when 1 < num_steps and (one or more than one) sampled indices exceeded buffer length
            boundary_index = rl.random.drng.choice(len(self._boundaries))
            if boundary_index != 0:
                boundary_low = self._boundaries[boundary_index - 1]
            else:
                boundary_low = 0
            boundary_high = self._boundaries[boundary_index]
            heap_index = rl.random.drng.integers(low=boundary_low, high=boundary_high)
            index = self._buffer.get_relative_index_from_heap_index(heap_index)
            if index < buffer_length - num_steps + 1:
                indices.append(index)
        return self.sample_indices(indices, num_steps)

    def update_priorities(self, errors: np.ndarray):
        errors = self._preprocess_errors(errors)
        indices = cast(Sequence[int], self._last_sampled_indices)
        for index, error in zip(indices, errors):
            self._buffer.update_priority(index, error)
        self._last_sampled_indices = None

    def _compute_segment_boundaries(self, N: int, k: int):
        if N < k:
            raise ValueError(f"Batch size {k} is greater than buffer size {N}")
        boundaries: List[int] = []
        denominator = self._ps_cumsum[N-1]
        for i in range(N):
            if (len(boundaries) + 1) / k <= self._ps_cumsum[i] / denominator:
                boundaries.append(i + 1)
        assert len(boundaries) == k
        return boundaries

    def _get_weights(self, indices: Sequence[int], alpha: float, beta: float):
        priorities = np.asarray([self._buffer.get_priority(i) for i in indices])[:, np.newaxis]
        worst_rank = len(self._buffer)
        min_priority = (1 / worst_rank) ** alpha
        return (priorities / min_priority) ** (-beta)


class PrioritizedReplayBuffer(ReplayBuffer):
    _variants: ClassVar[Sequence[str]] = ['proportional', 'rank_based']
    _buffer_impl: _PrioritizedReplayBuffer

    def __init__(self,
                 capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 betasteps: int = 10000,
                 error_clip: Optional[Tuple[float, float]] = (-1, 1),
                 epsilon: float = 1e-8,
                 reset_segment_interval: int = 1000,
                 sort_interval: int = 1000000,
                 variant: str = 'proportional'):
        if variant not in PrioritizedReplayBuffer._variants:
            raise ValueError(f'Unknown prioritized replay buffer variant: {variant}')
        if variant == 'proportional':
            self._buffer_impl = ProportionalPrioritizedReplayBuffer(capacity=capacity,
                                                                    alpha=alpha,
                                                                    beta=beta,
                                                                    betasteps=betasteps,
                                                                    error_clip=error_clip,
                                                                    epsilon=epsilon)
        elif variant == 'rank_based':
            self._buffer_impl = RankBasedPrioritizedReplayBuffer(capacity=capacity,
                                                                 alpha=alpha,
                                                                 beta=beta,
                                                                 betasteps=betasteps,
                                                                 error_clip=error_clip,
                                                                 reset_segment_interval=reset_segment_interval,
                                                                 sort_interval=sort_interval)
        else:
            raise NotImplementedError

    @property
    def capacity(self):
        return self._buffer_impl.capacity

    def append(self, experience):
        self._buffer_impl.append(experience)

    def append_all(self, experiences):
        self._buffer_impl.append_all(experiences)

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        return self._buffer_impl.sample(num_samples, num_steps)

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1):
        return self._buffer_impl.sample_indices(indices, num_steps)

    def update_priorities(self, errors: np.ndarray):
        self._buffer_impl.update_priorities(errors)

    def __len__(self):
        return len(self._buffer_impl)

    def __getitem__(self, item: int) -> Experience:
        return cast(Experience, self._buffer_impl[item])
