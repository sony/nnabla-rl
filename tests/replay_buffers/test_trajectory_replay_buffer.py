# Copyright 2023 Sony Group Corporation.
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

from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.trajectory_replay_buffer import TrajectoryReplayBuffer


class TestTrajectoryReplayBuffer():
    def test_len(self):
        trajectory_num = 10
        trajectory_length = 5
        buffer = self._generate_buffer_with_trajectories(trajectory_num=trajectory_num,
                                                         trajectory_length=trajectory_length)
        assert len(buffer) == trajectory_num * trajectory_length

    def test_trajectory_num(self):
        trajectory_num = 10
        trajectory_length = 5
        buffer = self._generate_buffer_with_trajectories(trajectory_num=trajectory_num,
                                                         trajectory_length=trajectory_length)
        assert buffer.trajectory_num == trajectory_num

    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_trajectories(trajectory_num=10)
        with pytest.raises(ValueError):
            buffer.sample(num_samples=100)

    def test_sample_trajectories(self):
        buffer = self._generate_buffer_with_trajectories(trajectory_num=10)
        trajectories, _ = buffer.sample_trajectories(num_samples=5)
        assert len(trajectories) == 5

    @pytest.mark.parametrize("num_samples", [i for i in range(1, 4)])
    @pytest.mark.parametrize("portion_length", [i for i in range(7, 11)])
    def test_sample_trajectories_portion(self, num_samples, portion_length):
        buffer = self._generate_buffer_with_trajectories(trajectory_num=10, trajectory_length=10)
        trajectories, _ = buffer.sample_trajectories_portion(num_samples=num_samples, portion_length=portion_length)
        assert len(trajectories) == num_samples
        assert all([len(trajectory) == portion_length for trajectory in trajectories])

    def test_sample_trajectories_portion_runtime_error(self):
        portion_length = 11
        buffer = self._generate_buffer_with_trajectories(trajectory_num=10, trajectory_length=10)
        with pytest.raises(RuntimeError):
            buffer.sample_trajectories_portion(num_samples=5, portion_length=portion_length)

    @pytest.mark.parametrize("portion_length", [i for i in range(1, 11)])
    def test_sample_indices_portion(self, portion_length):
        buffer = self._generate_buffer_with_trajectories(trajectory_num=10, trajectory_length=10)
        indices = [0, 9, 10, 19, 20, 29, 30, 39, 40, 45]
        trajectories, _ = buffer.sample_indices_portion(indices=indices, portion_length=portion_length)
        assert len(trajectories) == len(indices)
        assert all([len(trajectory) == portion_length for trajectory in trajectories])
        for i, index in enumerate(indices):
            trajectory_index = index // 10
            trajectory = buffer.get_trajectory(trajectory_index)
            index = index - 10 * trajectory_index
            start_index = min(index, len(trajectory) - portion_length)

            expected = trajectory[start_index:start_index+portion_length]
            actual = trajectories[i]
            assert len(expected) == len(actual)
            for expected_element, actual_element in zip(expected, actual):
                assert all([x is y for (x, y) in zip(expected_element, actual_element)])

    @pytest.mark.parametrize("num_samples", [i for i in range(1, 4)])
    @pytest.mark.parametrize("num_steps", [i for i in range(1, 4)])
    def test_sample(self, num_samples, num_steps):
        trajectory_num = 10
        buffer = self._generate_buffer_with_trajectories(trajectory_num=trajectory_num)

        samples, _ = buffer.sample(num_samples=num_samples, num_steps=num_steps)
        if num_steps == 1:
            assert num_samples == len(samples)
        else:
            assert num_steps == len(samples)
            assert num_samples == len(samples[0])

    def test_sample_indices(self):
        trajectory_num = 10

        trajectory_buffer = TrajectoryReplayBuffer()
        conventional_buffer = ReplayBuffer()
        for _ in range(trajectory_num):
            trajectory = self._generate_trajectory(trajectory_length=5)
            trajectory_buffer.append_trajectory(trajectory)
            conventional_buffer.append_all(trajectory)

        indices = np.random.randint(low=0, high=len(conventional_buffer), size=10)
        samples_from_trajectory_buffer, _ = trajectory_buffer.sample_indices(indices)
        samples_from_conventional_buffer, _ = conventional_buffer.sample_indices(indices)

        for (actual_sample, expected_sample) in zip(samples_from_trajectory_buffer, samples_from_conventional_buffer):
            for actual_item, expected_item in zip(actual_sample, expected_sample):
                np.testing.assert_almost_equal(actual_item, expected_item)

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

    def _generate_buffer_with_trajectories(self, trajectory_num, trajectory_length=5):
        buffer = TrajectoryReplayBuffer()
        for _ in range(trajectory_num):
            trajectory = self._generate_trajectory(trajectory_length=trajectory_length)
            buffer.append_trajectory(trajectory)
        return buffer

    def _generate_trajectory(self, trajectory_length):
        trajectory = []
        for _ in range(trajectory_length):
            experience = self._generate_experience_mock()
            trajectory.append(experience)
        return trajectory


if __name__ == "__main__":
    pytest.main()
