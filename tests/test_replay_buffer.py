# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import pytest

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer


class TestReplayBuffer(object):
    def test_infinite_capacity(self):
        buffer = ReplayBuffer()
        assert buffer.capacity is None

    def test_finite_capacity(self):
        requested_capacity = 100
        buffer = ReplayBuffer(capacity=requested_capacity)
        assert buffer.capacity == requested_capacity

    def test_append_without_capacity(self):
        append_num = 10
        buffer = ReplayBuffer()
        for i in range(append_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
            assert len(buffer) == i + 1

        assert len(buffer) == append_num

    def test_append_with_capacity(self):
        capacity = 5
        append_num = 10
        buffer = ReplayBuffer(capacity=capacity)
        for i in range(append_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
            if i < capacity:
                assert len(buffer) == i + 1
            else:
                assert len(buffer) == capacity
        assert len(buffer) == capacity

    def test_sample_one_experience(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        experience, info = buffer.sample()
        assert len(experience) == 1
        assert len(info["weights"]) == 1
        assert np.allclose(info["weights"], 1.0)

    def test_sample_multiple_experiences(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        experience, info = buffer.sample(num_samples=10)
        assert len(experience) == 10
        assert len(info["weights"]) == 10
        assert np.allclose(info["weights"], 1.0)

    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_experiences(experience_num=10)
        with pytest.raises(ValueError):
            buffer.sample(num_samples=100)

    def test_sample_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        indices = [1, 67, 50, 4, 99]

        experiences, info = buffer.sample_indices(indices)

        assert len(experiences) == len(indices)
        assert len(info["weights"]) == len(indices)
        assert np.allclose(info["weights"], 1.0)
        for experience, index in zip(experiences, indices):
            assert experience == buffer._buffer[index]

    def test_sample_from_empty_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        with pytest.raises(ValueError):
            buffer.sample_indices([])

    def test_sample_from_wrong_indices(self):
        buffer = self._generate_buffer_with_experiences(experience_num=100)
        indices = [-99, 100, 101]
        with pytest.raises(IndexError):
            buffer.sample_indices(indices)

    def test_random_indices(self):
        buffer = ReplayBuffer()
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
        buffer = ReplayBuffer()
        for _ in range(10):
            experience = self._generate_experience_mock()
            buffer.append(experience)

        assert len(buffer) == 10

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
        buffer = ReplayBuffer()
        for _ in range(experience_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
        return buffer


if __name__ == "__main__":
    pytest.main()
