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

from typing import Any, Dict, MutableSequence, Optional, Sequence, Tuple, Union, cast

import numpy as np

import nnabla_rl as rl
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience, Trajectory


class TrajectoryReplayBuffer(ReplayBuffer):
    """TrajectoryReplayBuffer.

    Enables appending/sampling not just an experience from the buffer
    but also a trajectory. In order to append/sample trajectory, the
    environment must be finite horizon setting.
    """

    def __init__(self, num_trajectories=None):
        super(TrajectoryReplayBuffer, self).__init__(num_trajectories)

        self._samples_per_trajectory = []
        self._num_experiences = 0  # this is num_trajectories * experiences_per_trajectory
        self._cumsum_experiences = [0]  # this is cummulative sum of self._num_experiences

    def __len__(self):
        return self._num_experiences

    def __getitem__(self, item: int) -> Experience:
        return self._get_experience(item)

    def get_trajectory(self, trajectory_index: int) -> Trajectory:
        return cast(Trajectory, self._buffer[trajectory_index])

    @property
    def trajectory_num(self):
        return len(self._samples_per_trajectory)

    def append(self, experience: Experience):
        raise NotImplementedError

    def append_all(self, experiences: Sequence[Experience]):
        raise NotImplementedError

    def append_trajectory(self, trajectory: Trajectory):
        self._buffer.append(trajectory)
        self._samples_per_trajectory.append(len(trajectory))
        num_experiences = 0
        cumsum_experiences = []
        for i in range(self.trajectory_num):
            num_experiences += self._samples_per_trajectory[i]
            cumsum_experiences.append(num_experiences)
        self._num_experiences = num_experiences
        self._cumsum_experiences = cumsum_experiences

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1) \
            -> Tuple[Union[Sequence[Experience], Tuple[Sequence[Experience], ...]], Dict[str, Any]]:
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        if num_steps < 1:
            raise ValueError(f'num_steps: {num_steps} should be greater than 0!')
        experiences: Union[Sequence[Experience], Tuple[Sequence[Experience], ...]]
        if num_steps == 1:
            experiences = [self._get_experience(index) for index in indices]
        else:
            experiences = tuple([self._get_experience(index+i) for index in indices] for i in range(num_steps))
        weights = np.ones([len(indices), 1])
        return experiences, dict(weights=weights)

    def sample_trajectories(self, num_samples: int = 1) -> Tuple[Union[Trajectory, Tuple[Trajectory, ...]],
                                                                 Dict[str, Any]]:
        """Randomly sample num_samples trajectories from the replay buffer.

        Args:
            num_samples (int): Number of samples to sample from the replay buffer. Defaults to 1.
        Returns:
            trajectories (Tuple[Trajectory, ...]): Randomly sampled num_samples of trajectories.
            info (Dict[str, Any]): dictionary of information about trajectories.
        Raises:
            ValueError: num_samples exceeds the maximum possible trajectories or num_steps is 0 or negative.
        """
        max_index = self.trajectory_num
        if num_samples > max_index:
            raise ValueError(
                f'num_samples: {num_samples} is greater than the number of trajectories saved in buffer: {max_index}')
        indices = self._random_trajectory_indices(num_samples=num_samples, max_index=max_index)
        return self.sample_indices_trajectory(indices)

    def sample_indices_trajectory(self, indices: Sequence[int]) \
            -> Tuple[Union[Trajectory, Tuple[Trajectory, ...]], Dict[str, Any]]:
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        trajectories: Union[Trajectory, Tuple[Trajectory, ...]]
        if len(indices) == 1:
            trajectories = self.get_trajectory(indices[0])
        else:
            trajectories = tuple(self.get_trajectory(index) for index in indices)
        weights = np.ones([len(indices), 1])
        return trajectories, dict(weights=weights)

    def sample_trajectories_portion(self,
                                    num_samples: int = 1,
                                    portion_length: int = 1) -> Tuple[Tuple[Trajectory, ...], Dict[str, Any]]:
        """Randomly sample num_samples trajectories with length portion_length
        from the replay buffer. (i.e. Each trajectory length will be
        portion_length) Trajectory will be sampled as follows. First, a
        trajectory will be sampled with probablity proportional to its length.

        Then, a random initial index between 0 to len(sampled_trajectory) - portion_length will be sampled.
        Finally, a portion_length size trajectory starting from sampled intial index will be returned.

        Args:
            num_samples (int): Number of samples to sample from the replay buffer. Defaults to 1.
            portion_length (int): Length of each sampled trajectory. Defaults to 1.

        Returns:
            trajectories (Tuple[Trajectory, ...]):
                Randomly sampled num_samples of trajectories with given portion_length.
            info (Dict[str, Any]): dictionary of information about trajectories.
        Raises:
            ValueError: num_samples exceeds the maximum possible trajectories or num_steps is 0 or negative.
            RuntimeError: Trajectory's length is below portion_length.
        """
        max_index = self.trajectory_num
        p = np.asarray(self._samples_per_trajectory)
        p = p / np.sum(p)
        trajectory_indices = rl.random.drng.choice(max_index, size=num_samples, replace=True, p=p)
        trajectories: MutableSequence[Trajectory] = [self.get_trajectory(index) for index in trajectory_indices]

        sliced_trajectories: MutableSequence[Trajectory] = []
        for trajectory in trajectories:
            max_index = len(trajectory)-portion_length
            if max_index < 0:
                raise RuntimeError(f'Trajectory length is shorter than portion length: {portion_length}')
            initial_index = rl.random.drng.choice(max_index+1, replace=False)
            sliced_trajectories.append(trajectory[initial_index:initial_index+portion_length])
        weights = np.ones([len(trajectories), 1])
        return tuple(sliced_trajectories), dict(weights=weights)

    def sample_indices_portion(self, indices: Sequence[int], portion_length: int = 1) ->  \
            Tuple[Tuple[Trajectory, ...], Dict[str, Any]]:
        """Sample trajectory portions from the buffer. (i.e. Each trajectory
        length will be portion_length) Trajectory from given index to
        index+portion_length-1 will be sampled. Index should be the index of a
        experience in the buffer and not trajectory's index. For example, if
        this buffer has 10 trajectories which consist of 5 experiences each,
        then: index 0: first experience of the first trajectory. index 1:
        second experience of the first trajectory. index 5: first experience of
        the second trajectory. index 6: second experience of the second
        trajectory. If index + portion_length exceeds the length of a
        trajectory, this will sample.

        from len(trajectory) - portion_length to len(trajectory) - 1

        Args:
            indices (int): Indices of the experience to sample from the replay buffer.
            portion_length (int): Length of each sampled trajectory. Defaults to 1.

        Returns:
            trajectories (Tuple[Trajectory, ...]):
                Randomly sampled num_samples of trajectories with given portion_length.
            info (Dict[str, Any]): dictionary of information about trajectories.
        Raises:
            RuntimeError: Trajectory's length is below portion_length.
        """
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        if portion_length < 1:
            raise ValueError(f'portion_length: {portion_length} should be greater than 0!')

        sliced_trajectories: MutableSequence[Trajectory] = []
        for index in indices:
            trajectory_index = np.argwhere(np.asarray(self._cumsum_experiences) > index)[0][0]
            trajectory = self.get_trajectory(trajectory_index)
            if len(trajectory) < portion_length:
                raise RuntimeError(f'Trajectory length is shorter than portion length: {portion_length}')

            if 0 < trajectory_index:
                experience_index = index - self._cumsum_experiences[trajectory_index - 1]
            else:
                experience_index = index
            experience_index = min(experience_index, len(trajectory) - portion_length)
            sliced_trajectories.append(trajectory[experience_index:experience_index+portion_length])

        weights = np.ones([len(indices), 1])
        return tuple(sliced_trajectories), dict(weights=weights)

    def _random_indices(self, num_samples: int, max_index: Optional[int] = None) -> Sequence[int]:
        if max_index is None:
            max_index = self._num_experiences
        # NOTE: Do NOT replace with np.random.choice(max_index, size=num_samples, replace=False)
        # np.random.choice is terribly slow when sampling without replacement
        indices = rl.random.drng.choice(max_index, size=num_samples, replace=False)
        return cast(Sequence[int], indices)

    def _random_trajectory_indices(self, num_samples: int, max_index: Optional[int] = None) -> Sequence[int]:
        if max_index is None:
            max_index = self.trajectory_num
        # NOTE: Do NOT replace with np.random.choice(max_index, size=num_samples, replace=False)
        # np.random.choice is terribly slow when sampling without replacement
        indices = rl.random.drng.choice(max_index, size=num_samples, replace=False)
        return cast(Sequence[int], indices)

    def _get_experience(self, experience_index) -> Experience:
        prev_cumsum = 0
        for trajectory_index, cumsum in enumerate(self._cumsum_experiences):
            if experience_index < cumsum:
                trajectory: Trajectory = self.get_trajectory(trajectory_index)
                experience: Experience = trajectory[experience_index - prev_cumsum]
                return experience
            prev_cumsum = cumsum
        raise ValueError(f'index {experience_index} is out of range')
