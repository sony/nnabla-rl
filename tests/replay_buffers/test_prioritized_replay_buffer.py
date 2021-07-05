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

import numpy as np
import pytest

from nnabla_rl.replay_buffers.prioritized_replay_buffer import (MaxHeap, MaxHeapDataHolder, MinTree,
                                                                ProportionalPrioritizedReplayBuffer,
                                                                RankBasedPrioritizedReplayBuffer, SumTree,
                                                                SumTreeDataHolder)


class TestMinTree(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = MinTree(capacity=requested_capacity)
        assert buffer._capacity == requested_capacity

    def test_append_with_capacity(self):
        capacity = 50
        append_num = 100
        min_tree = MinTree(capacity=capacity)
        for i in range(append_num):
            min_tree.append(i)
            if i < capacity:
                assert len(min_tree) == i + 1
                assert min_tree.min() == 0
            else:
                assert len(min_tree) == capacity
                assert min_tree.min() == i - capacity + 1
        assert len(min_tree) == capacity

    def test_update(self):
        capacity = 50
        min_tree = MinTree(capacity=capacity)
        priorities = np.random.randn(capacity)
        priorities = list(sorted(priorities))
        min_priority = math.inf
        for priority in priorities:
            min_priority = min(min_priority, priority)
            min_tree.append(priority)
            assert min_tree.min() == min_priority

        for i in range(capacity - 1):
            min_tree.update(i, math.inf)
            assert min_tree.min() == priorities[i + 1]


class TestSumTree(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = SumTree(capacity=requested_capacity)
        assert buffer._capacity == requested_capacity

    def test_append_with_capacity(self):
        capacity = 50
        append_num = 100
        sum_tree = SumTree(capacity=capacity)
        expected_sum = 0
        for i in range(append_num):
            sum_tree.append(i)
            if i < capacity:
                assert len(sum_tree) == i + 1
            else:
                expected_sum += i
                assert len(sum_tree) == capacity
        assert len(sum_tree) == capacity
        assert sum_tree.sum() == expected_sum

    def test_update(self):
        capacity = 50
        sum_tree = SumTree(capacity=capacity)

        priorities = np.random.randn(capacity)
        priorities = list(sorted(priorities))
        sum = 0.0
        for priority in priorities:
            sum += priority
            sum_tree.append(priority)
            np.testing.assert_almost_equal(sum_tree.sum(), sum)

        for i in range(capacity - 1):
            tree_index = sum_tree.absolute_to_tree_index(i)
            current_priority = sum_tree[tree_index]
            new_priority = np.random.randn()
            sum_tree.update(i, new_priority)
            sum += new_priority - current_priority
            np.testing.assert_almost_equal(sum_tree.sum(), sum)


class TestMaxHeap(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = MaxHeap(capacity=requested_capacity)
        assert buffer._capacity == requested_capacity

    def test_append_more_than_capacity(self):
        capacity = 10
        max_heap = MaxHeap(capacity=capacity)
        # Fill heap
        for i in range(capacity):
            max_heap.append(i)
            assert max_heap[0][1] == i

        # Fill heap
        for i in range(5):
            max_heap.append(i + len(max_heap))
            # Check appended data is on the top
            assert max_heap[0][1] == i + len(max_heap)

    def test_update_priorities(self):
        capacity = 10
        max_heap = MaxHeap(capacity=capacity)
        # create positive random priorities to easily manipulate the top priority of heap by inverting the sign
        priorities = np.abs(np.random.randn(capacity))
        max_priority = -math.inf
        for priority in priorities:
            max_priority = max(max_priority, priority)
            max_heap.append(priority)
            assert max_heap[0][1] == max_priority
        sorted_priority = list(reversed(sorted(priorities)))

        # update data priorities
        for i in range(capacity - 1):
            current_priority = max_heap[0][1]
            # All priorities are positive -> this will be removed from the top
            new_priority = -current_priority
            absolute_index = max_heap.heap_to_absolute_index(0)
            max_heap.update(absolute_index, new_priority)
            assert max_heap[0][1] == sorted_priority[i + 1]

    def test_sort_data(self):
        capacity = 10
        max_heap = MaxHeap(capacity=capacity)
        priorities = np.random.randn(capacity)
        max_priority = -math.inf
        for priority in priorities:
            max_priority = max(max_priority, priority)
            max_heap.append(priority)
            assert max_heap[0][1] == max_priority

        # should not be ideally sorted
        sorted_order = list(reversed(sorted(priorities)))
        actual_order = [max_heap._heap[i][1] for i in range(capacity)]
        assert not np.allclose(actual_order, sorted_order)

        max_heap.sort_data()
        actual_order = [max_heap._heap[i][1] for i in range(capacity)]
        assert np.allclose(actual_order, sorted_order)


class TestSumTreeDataHolder(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_get_priority(self):
        capacity = 10
        initial_max_priority = 1.0
        holder = SumTreeDataHolder(capacity=capacity, initial_max_priority=initial_max_priority)

        indices = range(10)
        for i in indices:
            # save index as data
            holder.append(i)

        for i in indices:
            actual_priority = holder.get_priority(i)
            # Initial priority is initial_max_priority
            assert initial_max_priority == actual_priority

        priorities = [np.random.uniform() for i in range(10)]
        for index, priority in zip(indices, priorities):
            holder.update_priority(index, priority)

        for index, expected_priority in zip(indices, priorities):
            actual_priority = holder.get_priority(index)
            # Initial priority is initial_max_priority
            assert expected_priority == actual_priority

    def test_sum_priority(self):
        capacity = 10
        initial_max_priority = 1.0
        holder = SumTreeDataHolder(capacity=capacity, initial_max_priority=initial_max_priority)

        indices = range(10)
        for i in indices:
            # save index as data
            holder.append(i)

        np.testing.assert_almost_equal(holder.sum_priority(), initial_max_priority * len(indices))

        priorities = [np.random.uniform() for i in range(10)]
        for index, priority in zip(indices, priorities):
            holder.update_priority(index, priority)

        np.testing.assert_almost_equal(holder.sum_priority(), np.sum(priorities))

    def test_min_priority(self):
        capacity = 10
        initial_max_priority = 1.0
        holder = SumTreeDataHolder(capacity=capacity, initial_max_priority=initial_max_priority)

        indices = range(10)
        for i in indices:
            # save index as data
            holder.append(i)

        np.testing.assert_almost_equal(holder.min_priority(), initial_max_priority)

        priorities = [np.random.uniform() for i in range(10)]
        for index, priority in zip(indices, priorities):
            holder.update_priority(index, priority)

        np.testing.assert_almost_equal(holder.min_priority(), np.min(priorities))


class TestMaxHeapDataHolder(object):
    def setup_method(self, method):
        np.random.seed(0)

    @pytest.mark.parametrize("alpha", [np.random.random() for _ in range(1, 5)])
    def test_compute_priority(self, alpha):
        capacity = 10
        holder = MaxHeapDataHolder(capacity=capacity, alpha=alpha)

        for rank in range(1, 10):
            expected = (1 / rank) ** alpha
            actual = holder._compute_priority(rank)
            assert np.allclose(expected, actual)


class TestProportionalPrioritizedReplayBuffer(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = ProportionalPrioritizedReplayBuffer(capacity=requested_capacity)
        assert buffer.capacity == requested_capacity

    def test_infine_capacity(self):
        with pytest.raises(ValueError):
            _ = ProportionalPrioritizedReplayBuffer(capacity=None)

    def test_non_positive_capacity(self):
        with pytest.raises(ValueError):
            _ = ProportionalPrioritizedReplayBuffer(capacity=-1)
        with pytest.raises(ValueError):
            _ = ProportionalPrioritizedReplayBuffer(capacity=0)

    def test_append_with_capacity(self):
        capacity = 5
        append_num = 10
        buffer = ProportionalPrioritizedReplayBuffer(capacity=capacity)
        for i in range(append_num):
            experience = _generate_experience_mock()
            buffer.append(experience)
            if i < capacity:
                assert len(buffer) == i + 1
            else:
                assert len(buffer) == capacity
        assert len(buffer) == capacity

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 5)])
    @pytest.mark.parametrize("betasteps", [i for i in range(1, 5)])
    def test_betasteps(self, beta, betasteps):
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta, betasteps=betasteps)
        _, _ = buffer.sample()
        new_beta = beta + (1.0 - beta) / betasteps
        np.testing.assert_almost_equal(buffer._beta, new_beta)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    @pytest.mark.parametrize("normalization_method", ["batch_max", "buffer_max"])
    def test_sample_one_experience(self, beta, normalization_method):
        buffer = self._generate_buffer_with_experiences(
            experience_num=100, beta=beta, normalization_method=normalization_method)
        experiences, info = buffer.sample()
        indices = buffer._last_sampled_indices
        assert len(experiences) == 1
        assert "weights" in info
        assert len(info["weights"]) == 1
        assert len(indices) == 1
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    @pytest.mark.parametrize("normalization_method", ["batch_max", "buffer_max"])
    def test_sample_multiple_experiences(self, beta, normalization_method):
        buffer = self._generate_buffer_with_experiences(
            experience_num=100, beta=beta, normalization_method=normalization_method)
        num_samples = 10
        experiences, info = buffer.sample(num_samples=num_samples)
        indices = buffer._last_sampled_indices
        assert len(experiences) == num_samples
        assert "weights" in info
        assert len(info["weights"]) == num_samples
        assert len(indices) == num_samples
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    @pytest.mark.parametrize("num_steps", range(1, 5))
    @pytest.mark.parametrize("normalization_method", ["batch_max", "buffer_max"])
    def test_sample_multiple_step_experience(self, beta, num_steps, normalization_method):
        buffer = self._generate_buffer_with_experiences(
            experience_num=100, beta=beta, normalization_method=normalization_method)
        experiences_tuple, info = buffer.sample(num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = tuple([experiences_tuple, ])
        indices = buffer._last_sampled_indices
        assert len(experiences_tuple) == num_steps
        assert "weights" in info
        assert len(info["weights"]) == 1
        assert len(indices) == 1
        weights = info["weights"]
        for i, experiences in enumerate(experiences_tuple):
            for index, experience, actual_weight in zip(indices, experiences, weights):
                expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
                actual_weight = actual_weight[0]
                assert experience == buffer[index + i]
                np.testing.assert_almost_equal(expected_weight, actual_weight)

    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_experiences(experience_num=10)
        with pytest.raises(ValueError):
            buffer.sample(num_samples=100)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    @pytest.mark.parametrize("normalization_method", ["batch_max", "buffer_max"])
    def test_sample_indices(self, beta, normalization_method):
        buffer = self._generate_buffer_with_experiences(
            experience_num=100, beta=beta, normalization_method=normalization_method)
        indices = [1, 67, 50, 4, 99]

        experiences, info = buffer.sample_indices(indices)

        assert len(experiences) == len(indices)
        assert "weights" in info
        assert len(info["weights"]) == len(indices)
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    def test_sample_from_empty_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        with pytest.raises(ValueError):
            buffer.sample_indices([])

    def test_sample_from_wrong_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        indices = [-99, 100, 101]
        with pytest.raises(KeyError):
            buffer.sample_indices(indices)

    def test_random_indices(self):
        buffer = ProportionalPrioritizedReplayBuffer(capacity=100)
        for _ in range(100):
            experience = _generate_experience_mock()
            buffer.append(experience)

        indices = buffer._random_indices(num_samples=10)
        indices = np.array(indices, dtype=np.int32)
        assert len(indices) == 10
        assert np.alltrue(0 <= indices) and np.alltrue(indices <= 100)

        # check no duplicates
        assert len(np.unique(indices)) == len(indices)

    def test_buffer_len(self):
        buffer = ProportionalPrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            experience = _generate_experience_mock()
            buffer.append(experience)

        assert len(buffer) == 10

    def test_sample_without_update(self):
        beta = 0.5
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        indices = [1, 67, 50, 4, 99]
        _, weights = buffer.sample_indices(indices)

        # update the priority and check that following sampling succeeds
        errors = np.random.sample([len(weights), 1])
        buffer.update_priorities(errors)

        _, _ = buffer.sample_indices(indices)

        # sample without priority update
        with pytest.raises(RuntimeError):
            buffer.sample_indices(indices)

    def test_update_priorities(self):
        beta = 0.5
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        indices = [1, 67, 50, 4, 99]
        _, weights = buffer.sample_indices(indices)

        # update the priority and check that following sampling succeeds
        errors = np.random.uniform(size=(len(indices), 1))
        buffer.update_priorities(errors)

        expected_priorities = (np.abs(errors) + 1e-8) ** buffer._alpha
        for index, expected_priority in zip(indices, expected_priorities):
            actual_priority = buffer._buffer.get_priority(index)
            np.testing.assert_almost_equal(expected_priority, actual_priority)

        new_indices = [0, 2, 49, 51, 3, 5, 98]
        _, weights = buffer.sample_indices(new_indices)
        errors = np.random.uniform(size=(len(new_indices), 1))
        buffer.update_priorities(errors)

        expected_new_priorities = (np.abs(errors) + 1e-8) ** buffer._alpha
        for index, expected_priority in zip(new_indices, expected_new_priorities):
            actual_priority = buffer._buffer.get_priority(index)
            np.testing.assert_almost_equal(expected_priority, actual_priority)

        # Check old priorities still not changed
        for index, expected_priority in zip(indices, expected_priorities):
            # New data is appended -> previous indices - 1 is the correct index
            actual_priority = buffer._buffer.get_priority(index)
            np.testing.assert_almost_equal(expected_priority, actual_priority)

        # Append new data and check index 0's priority has changed
        experience = self._generate_experience_mock()
        buffer.append(experience)

        old_index0_priority = expected_new_priorities[0]
        new_index0_priority = buffer._buffer.get_priority(0)
        with pytest.raises(AssertionError):
            np.testing.assert_almost_equal(old_index0_priority, new_index0_priority)

        # Check again old priorities still not changed
        for index, expected_priority in zip(indices, expected_priorities):
            index = index - 1
            actual_priority = buffer._buffer.get_priority(index)
            np.testing.assert_almost_equal(expected_priority, actual_priority)

    @pytest.mark.parametrize("error_clip", [(-np.random.uniform(), np.random.uniform()) for _ in range(1, 10)])
    def test_error_preprocessing(self, error_clip):
        buffer = ProportionalPrioritizedReplayBuffer(capacity=100, error_clip=error_clip)

        batch_size = 10
        errors = np.random.randn(batch_size, 1)
        processed = buffer._preprocess_errors(errors)

        max_error = np.float32(max(np.abs(error_clip[0]), error_clip[1])) + 1e-5
        assert all(0 <= processed)
        assert all(max_error >= processed)

    def _generate_experience_mock(self):
        state_shape = (5, )
        action_shape = (10, )

        state = np.empty(shape=state_shape)
        action = np.empty(shape=action_shape)
        reward = np.random.normal()
        non_terminal = 0.0 if np.random.choice([True, False], 1) else 1.0
        next_state = np.empty(shape=state_shape)
        next_action = np.empty(shape=action_shape)

        return (state, action, reward, non_terminal, next_state, next_action)

    def _generate_buffer_with_experiences(self,
                                          experience_num, beta=1.0,
                                          betasteps=1,
                                          normalization_method="batch_max"):
        buffer = ProportionalPrioritizedReplayBuffer(
            capacity=experience_num, beta=beta, betasteps=betasteps, normalization_method=normalization_method)
        for _ in range(experience_num):
            experience = _generate_experience_mock()
            buffer.append(experience)
        return buffer

    def _compute_weight(self, buffer, index, alpha, beta):
        priority = buffer._buffer.get_priority(index)
        if buffer._normalization_method == "batch_max":
            min_priority = np.min(np.array([buffer._buffer.get_priority(index)
                                            for index in buffer._last_sampled_indices]))
        elif buffer._normalization_method == "buffer_max":
            min_priority = buffer._buffer.min_priority()
        else:
            raise RuntimeError
        weights = (priority / min_priority) ** (-beta)
        return weights


class TestRankBasedPrioritizedReplayBuffer(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = RankBasedPrioritizedReplayBuffer(capacity=requested_capacity)
        assert buffer.capacity == requested_capacity

    def test_infine_capacity(self):
        with pytest.raises(ValueError):
            _ = RankBasedPrioritizedReplayBuffer(capacity=None)

    def test_non_positive_capacity(self):
        with pytest.raises(ValueError):
            _ = RankBasedPrioritizedReplayBuffer(capacity=-1)
        with pytest.raises(ValueError):
            _ = RankBasedPrioritizedReplayBuffer(capacity=0)

    def test_append_with_capacity(self):
        capacity = 5
        append_num = 10
        buffer = RankBasedPrioritizedReplayBuffer(capacity=capacity)
        for i in range(append_num):
            experience = _generate_experience_mock()
            buffer.append(experience)
            if i < capacity:
                assert len(buffer) == i + 1
            else:
                assert len(buffer) == capacity
        assert len(buffer) == capacity

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 5)])
    @pytest.mark.parametrize("betasteps", [i for i in range(1, 5)])
    def test_betasteps(self, beta, betasteps):
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta, betasteps=betasteps)
        _, _ = buffer.sample()
        new_beta = beta + (1.0 - beta) / betasteps
        np.testing.assert_almost_equal(buffer._beta, new_beta)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    def test_sample_one_experience(self, beta):
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        experiences, info = buffer.sample()
        indices = buffer._last_sampled_indices
        assert len(experiences) == 1
        assert "weights" in info
        assert len(info["weights"]) == 1
        assert len(indices) == 1
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    def test_sample_multiple_experiences(self, beta):
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        num_samples = 10
        experiences, info = buffer.sample(num_samples=num_samples)
        indices = buffer._last_sampled_indices
        assert len(experiences) == num_samples
        assert "weights" in info
        assert len(info["weights"]) == num_samples
        assert len(indices) == num_samples
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    @pytest.mark.parametrize("num_steps", range(1, 5))
    def test_sample_multiple_step_experience(self, beta, num_steps):
        buffer = self._generate_buffer_with_experiences(experience_num=100,
                                                        beta=beta)
        experiences_tuple, info = buffer.sample(num_steps=num_steps)
        if num_steps == 1:
            experiences_tuple = tuple([experiences_tuple, ])
        indices = buffer._last_sampled_indices
        assert len(experiences_tuple) == num_steps
        assert "weights" in info
        assert len(info["weights"]) == 1
        assert len(indices) == 1
        weights = info["weights"]
        for i, experiences in enumerate(experiences_tuple):
            for index, experience, actual_weight in zip(indices, experiences, weights):
                expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
                actual_weight = actual_weight[0]
                assert experience == buffer[index + i]
                np.testing.assert_almost_equal(expected_weight, actual_weight)

    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_experiences(experience_num=10)
        with pytest.raises(ValueError):
            buffer.sample(num_samples=100)

    @pytest.mark.parametrize("beta", [np.random.uniform(low=0.0, high=1.0) for _ in range(1, 10)])
    def test_sample_indices(self, beta):
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        indices = [1, 67, 50, 4, 99]

        experiences, info = buffer.sample_indices(indices)

        assert len(experiences) == len(indices)
        assert "weights" in info
        assert len(info["weights"]) == len(indices)
        weights = info["weights"]
        for index, experience, actual_weight in zip(indices, experiences, weights):
            expected_weight = self._compute_weight(buffer, index, alpha=buffer._alpha, beta=beta)
            actual_weight = actual_weight[0]
            assert experience == buffer[index]
            np.testing.assert_almost_equal(expected_weight, actual_weight)

    def test_sample_from_empty_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        with pytest.raises(ValueError):
            buffer.sample_indices([])

    def test_sample_from_wrong_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        indices = [-99, 100, 101]
        with pytest.raises(KeyError):
            buffer.sample_indices(indices)

    def test_random_indices(self):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=100)
        for _ in range(100):
            experience = _generate_experience_mock()
            buffer.append(experience)

        indices = buffer._random_indices(num_samples=10)
        indices = np.array(indices, dtype=np.int32)
        assert len(indices) == 10
        assert np.alltrue(0 <= indices) and np.alltrue(indices <= 100)

        # check no duplicates
        assert len(np.unique(indices)) == len(indices)

    def test_buffer_len(self):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            experience = _generate_experience_mock()
            buffer.append(experience)

        assert len(buffer) == 10

    def test_sample_without_update(self):
        beta = 0.5
        buffer = self._generate_buffer_with_experiences(experience_num=100, beta=beta)
        indices = [1, 67, 50, 4, 99]
        _, weights = buffer.sample_indices(indices)

        # update the priority and check that following sampling succeeds
        errors = np.random.sample([len(weights), 1])
        buffer.update_priorities(errors)

        _, _ = buffer.sample_indices(indices)

        # sample without priority update
        with pytest.raises(RuntimeError):
            buffer.sample_indices(indices)

    @pytest.mark.parametrize("N, k", [(1, 1), (2, 1), (3, 2), (4, 4), (10, 5)])
    def test_compute_segment_boundaries(self, N, k):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=10)
        actual = buffer._compute_segment_boundaries(N, k)

        assert len(actual) == k
        assert actual[-1] == N

    def test_compute_segment_boundaries_batch_size_greater_than_buffer(self):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=10)

        with pytest.raises(ValueError):
            buffer._compute_segment_boundaries(N=5, k=10)

    def test_sort_interval(self):
        sort_interval = 5
        buffer = RankBasedPrioritizedReplayBuffer(capacity=10, sort_interval=sort_interval)
        for i in range(10):
            experience = _generate_experience_mock()
            buffer.append(experience)
            buffer._last_sampled_indices = [i]
            buffer.update_priorities(errors=[np.random.randint(100)])
            if (i + 1) % sort_interval == 0:
                sorted_heap = sorted(buffer._buffer._max_heap._heap,
                                     key=lambda item: -math.inf if item is None else item[1],
                                     reverse=True)
                assert np.alltrue(buffer._buffer._max_heap._heap == sorted_heap)

    @pytest.mark.parametrize("error_clip", [(-np.random.uniform(), np.random.uniform()) for _ in range(1, 10)])
    def test_error_preprocessing(self, error_clip):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=100, error_clip=error_clip)

        batch_size = 10
        errors = np.random.randn(batch_size, 1)
        processed = buffer._preprocess_errors(errors)

        max_error = np.float32(max(np.abs(error_clip[0]), error_clip[1])) + 1e-5
        assert all(0 <= processed)
        assert all(max_error >= processed)

    def _generate_buffer_with_experiences(self, experience_num, beta=1.0, betasteps=1):
        buffer = RankBasedPrioritizedReplayBuffer(capacity=experience_num, beta=beta, betasteps=betasteps)
        for _ in range(experience_num):
            experience = _generate_experience_mock()
            buffer.append(experience)
        return buffer

    def _compute_weight(self, buffer, index, alpha, beta):
        priority = buffer._buffer.get_priority(index)
        worst_rank = len(buffer._buffer)
        min_priority = (1 / worst_rank) ** alpha
        weights = (priority / min_priority) ** (-beta)
        return weights


def _generate_experience_mock():
    state_shape = (5, )
    action_shape = (10, )

    state = np.empty(shape=state_shape)
    action = np.empty(shape=action_shape)
    reward = np.random.normal()
    non_terminal = 0.0 if np.random.choice([True, False], 1) else 1.0
    next_state = np.empty(shape=state_shape)
    next_action = np.empty(shape=action_shape)

    return (state, action, reward, non_terminal, next_state, next_action)


if __name__ == "__main__":
    pytest.main()
