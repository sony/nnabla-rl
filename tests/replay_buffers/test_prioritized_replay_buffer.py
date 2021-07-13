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
import pytest

from nnabla_rl.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer, SumTree


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
        init_max_p = 1.0
        buffer = SumTree(capacity=capacity, init_max_p=init_max_p)
        for i in range(append_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
            if i < capacity:
                assert len(buffer) == i + 1
            else:
                assert len(buffer) == capacity
        assert len(buffer) == capacity
        assert buffer._min_p == init_max_p
        assert buffer.total == init_max_p * capacity

    @pytest.mark.parametrize(
        "index, priority",
        [(np.random.randint(100), np.random.uniform(low=0.1, high=5.0))
         for _ in range(1, 10)])
    def test_update(self, index, priority):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        assert buffer.total == 100.0
        tree_index = index + buffer._capacity - 1
        prev_priority = buffer._tree[tree_index].value
        buffer._update(absolute_index=index, p=priority)
        assert buffer._tree[tree_index].value == priority
        assert buffer.total == 100.0 + (priority - prev_priority)

    @pytest.mark.parametrize("beta, priority",
                             [(np.random.randint(low=0.0, high=1.0),
                               np.random.uniform(low=0.1, high=5.0))
                              for _ in range(1, 10)])
    def test_weights_from_priorities(self, beta, priority):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        min_p = buffer._min_p
        weights = buffer.weights_from_priorities(priorities=np.array([priority]), beta=beta)
        assert weights == ((priority / min_p) ** (-beta))

    def test_relative_absolute_index_conversion(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)

        relative_indices = np.random.randint(low=0, high=100, size=10)
        absolute_indices = [buffer._relative_to_absolute_index(relative_index) for relative_index in relative_indices]

        reconstructed_indices = \
            [buffer._absolute_to_relative_index(absolute_index) for absolute_index in absolute_indices]
        assert all(relative_indices == reconstructed_indices)

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

    def _generate_buffer_with_experiences(self, experience_num):
        buffer = SumTree(capacity=experience_num, init_max_p=1.0)
        for _ in range(experience_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
        return buffer


class TestPrioritizedReplayBuffer(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = PrioritizedReplayBuffer(capacity=requested_capacity)
        assert buffer.capacity == requested_capacity

    def test_infine_capacity(self):
        with pytest.raises(ValueError):
            _ = PrioritizedReplayBuffer(capacity=None)

    def test_not_positive_capacity(self):
        with pytest.raises(ValueError):
            _ = PrioritizedReplayBuffer(capacity=-1)
        with pytest.raises(ValueError):
            _ = PrioritizedReplayBuffer(capacity=0)

    def test_append_with_capacity(self):
        capacity = 5
        append_num = 10
        buffer = PrioritizedReplayBuffer(capacity=capacity)
        for i in range(append_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
            if i < capacity:
                assert len(buffer) == i + 1
            else:
                assert len(buffer) == capacity
        assert len(buffer) == capacity

    @pytest.mark.parametrize(
        "beta",
        [np.random.randint(low=0.0, high=1.0) for _ in range(1, 10)])
    def test_sample_one_experience(self, beta):
        buffer = self._generate_buffer_with_experiences(experience_num=100,
                                                        beta=beta)
        experiences, info = buffer.sample()
        indices = buffer._last_sampled_indices
        assert len(experiences) == 1
        assert "weights" in info
        assert len(info["weights"]) == 1
        assert len(indices) == 1
        weights = info["weights"]
        for index, experience, weight in zip(indices, experiences, weights):
            priority = buffer._buffer.get_priority(index)
            assert experience == buffer[index]
            assert weight == buffer._buffer.weights_from_priorities(priority, beta)

    @pytest.mark.parametrize(
        "beta",
        [np.random.randint(low=0.0, high=1.0) for _ in range(1, 10)])
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
        for index, experience, weight in zip(indices, experiences, weights):
            priority = buffer._buffer.get_priority(index)
            assert experience == buffer[index]
            assert weight == buffer._buffer.weights_from_priorities(priority, beta)

    @pytest.mark.parametrize("beta", [np.random.randint(low=0.0, high=1.0) for _ in range(1, 5)])
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
            for index, experience, weight in zip(indices, experiences, weights):
                priority = buffer._buffer.get_priority(index)
                assert weight == buffer._buffer.weights_from_priorities(priority, beta)
                assert experience == buffer[index + i]

    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_experiences(experience_num=10)
        with pytest.raises(ValueError):
            buffer.sample(num_samples=100)

    @pytest.mark.parametrize(
        "beta",
        [np.random.randint(low=0.0, high=1.0) for _ in range(1, 10)])
    def test_sample_indices(self, beta):
        buffer = self._generate_buffer_with_experiences(experience_num=100,
                                                        beta=beta)
        indices = [1, 67, 50, 4, 99]

        experiences, info = buffer.sample_indices(indices)

        assert len(experiences) == len(indices)
        assert "weights" in info
        assert len(info["weights"]) == len(indices)
        weights = info["weights"]
        for index, experience, weight in zip(indices, experiences, weights):
            priority = buffer._buffer.get_priority(index)
            assert experience == buffer[index]
            assert weight == buffer._buffer.weights_from_priorities(priority, beta)

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
        buffer = PrioritizedReplayBuffer(capacity=100)
        for _ in range(100):
            experience = self._generate_experience_mock()
            buffer.append(experience)

        indices = buffer._random_indices(num_samples=10)
        indices = np.array(indices, dtype=np.int32)
        assert len(indices) == 10
        assert np.alltrue(0 <= indices) and np.alltrue(indices <= 100)

        # check no duplicates
        assert len(np.unique(indices)) == len(indices)

    def test_buffer_len(self):
        buffer = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            experience = self._generate_experience_mock()
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

    def _generate_buffer_with_experiences(self, experience_num, beta=1.0):
        buffer = PrioritizedReplayBuffer(capacity=experience_num, beta=beta)
        for _ in range(experience_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
        return buffer


if __name__ == "__main__":
    pytest.main()
