# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from collections import deque
from typing import Optional, Sequence, Tuple, cast

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers.prioritized_replay_buffer import (ProportionalPrioritizedReplayBuffer,
                                                                RankBasedPrioritizedReplayBuffer)
from nnabla_rl.replay_buffers.trajectory_replay_buffer import TrajectoryReplayBuffer
from nnabla_rl.typing import Trajectory
from nnabla_rl.utils.data import RingBuffer


class MemoryEfficientAtariBuffer(ReplayBuffer):
    """Buffer designed to compactly save experiences of Atari environments used
    in DQN.

    DQN (and other training algorithms) requires large replay buffer
    when training on Atari games. If you naively save the experiences,
    you'll need more than 100GB to save them (assuming 1M experiences).
    Which usually does not fit in the machine's memory (unless you have
    money:). This replay buffer reduces the size of experience by
    casting the images to uint8 and removing old frames concatenated to
    the observation. By using this buffer, you can hold 1M experiences
    using only 20GB(approx.) of memory. Note that this class is designed
    only for DQN style training on atari environment. (i.e. State
    consists of "stacked_frames" number of concatenated grayscaled
    frames and its values are normalized between 0 and 1)
    """
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _buffer: RingBuffer
    _sub_buffer: deque

    def __init__(self, capacity: int, stacked_frames: int = 4):
        super(MemoryEfficientAtariBuffer, self).__init__(capacity=capacity)
        self._reset = True
        self._buffer = RingBuffer(maxlen=capacity)
        self._sub_buffer = deque(maxlen=stacked_frames-1)
        self._stacked_frames = stacked_frames

    def append(self, experience):
        self._reset = _append_to_buffer(experience, self._buffer, self._sub_buffer, self._reset)

    def __getitem__(self, index: int):
        return _getitem_from_buffer(index, self._buffer, self._sub_buffer, self._stacked_frames)


class _LazyAtariTrajectory(object):
    def __init__(self, buffer: MemoryEfficientAtariBuffer):
        self._buffer = buffer

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._buffer[i] for i in range(key.stop)[key]]
        elif isinstance(key, int):
            return self._buffer[key]
        else:
            raise TypeError('Invalid key type')


class MemoryEfficientAtariTrajectoryBuffer(TrajectoryReplayBuffer):
    def __init__(self, num_trajectories=None):
        super(MemoryEfficientAtariTrajectoryBuffer, self).__init__(num_trajectories)

    def append_trajectory(self, trajectory: Trajectory):
        # Use memory efficient atari buffer to save the trajectory efficiently
        atari_buffer = MemoryEfficientAtariBuffer(capacity=len(trajectory))
        atari_buffer.append_all(trajectory)

        self._buffer.append(atari_buffer)

        # Below is the same as super class' code
        self._samples_per_trajectory.append(len(trajectory))
        num_experiences = 0
        cumsum_experiences = []
        for i in range(self.trajectory_num):
            num_experiences += self._samples_per_trajectory[i]
            cumsum_experiences.append(num_experiences)
        self._num_experiences = num_experiences
        self._cumsum_experiences = cumsum_experiences

    def get_trajectory(self, trajectory_index: int) -> Trajectory:
        return self._buffer_to_trajectory(self._get_atari_buffer(trajectory_index))

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        raise NotImplementedError

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1):
        raise NotImplementedError

    def _buffer_to_trajectory(self, buffer: MemoryEfficientAtariBuffer) -> Trajectory:
        return cast(Trajectory, _LazyAtariTrajectory(buffer))

    def _get_atari_buffer(self, trajectory_index: int) -> MemoryEfficientAtariBuffer:
        return cast(MemoryEfficientAtariBuffer, self._buffer[trajectory_index])


class ProportionalPrioritizedAtariBuffer(ProportionalPrioritizedReplayBuffer):
    """Prioritized buffer designed to compactly save experiences of Atari
    environments used in DQN.

    Proportional Prioritized version of efficient Atari buffer. Note
    that this class is designed only for DQN style training on atari
    environment. (i.e. State consists of "stacked_frames" number of
    concatenated grayscaled frames and its values are normalized between
    0 and 1)
    """
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _sub_buffer: deque

    def __init__(self,
                 capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 betasteps: int = 50000000,
                 error_clip: Optional[Tuple[float, float]] = (-1, 1),
                 epsilon: float = 1e-8,
                 normalization_method: str = "buffer_max",
                 stacked_frames: int = 4):
        super(ProportionalPrioritizedAtariBuffer, self).__init__(capacity=capacity,
                                                                 alpha=alpha,
                                                                 beta=beta,
                                                                 betasteps=betasteps,
                                                                 error_clip=error_clip,
                                                                 epsilon=epsilon,
                                                                 normalization_method=normalization_method)
        self._reset = True
        self._sub_buffer = deque(maxlen=stacked_frames-1)
        self._stacked_frames = stacked_frames

    def append(self, experience):
        self._reset = _append_to_buffer(experience, self._buffer, self._sub_buffer, self._reset)

    def __getitem__(self, index: int):
        return _getitem_from_buffer(index, self._buffer, self._sub_buffer, self._stacked_frames)


class RankBasedPrioritizedAtariBuffer(RankBasedPrioritizedReplayBuffer):
    """Prioritized buffer designed to compactly save experiences of Atari
    environments used in DQN.

    RankBased Prioritized version of efficient Atari buffer. Note that
    this class is designed only for DQN style training on atari
    environment. (i.e. State consists of "stacked_frames" number of
    concatenated grayscaled frames and its values are normalized between
    0 and 1)
    """
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _sub_buffer: deque

    def __init__(self,
                 capacity: int,
                 alpha: float = 0.7,
                 beta: float = 0.5,
                 betasteps: int = 50000000,
                 error_clip: Optional[Tuple[float, float]] = (-1, 1),
                 reset_segment_interval: int = 1000,
                 sort_interval: int = 1000000,
                 stacked_frames: int = 4):
        super(RankBasedPrioritizedAtariBuffer, self).__init__(capacity=capacity,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              betasteps=betasteps,
                                                              error_clip=error_clip,
                                                              reset_segment_interval=reset_segment_interval,
                                                              sort_interval=sort_interval)
        self._reset = True
        self._sub_buffer = deque(maxlen=stacked_frames-1)
        self._stacked_frames = stacked_frames

    def append(self, experience):
        self._reset = _append_to_buffer(experience, self._buffer, self._sub_buffer, self._reset)

    def __getitem__(self, index: int):
        return _getitem_from_buffer(index, self._buffer, self._sub_buffer, self._stacked_frames)


def _denormalize_state(state, scalar=255.0):
    return (state * scalar).astype(np.uint8)


def _normalize_state(state, scalar=255.0):
    return state.astype(np.float32) / scalar


def _is_float(state):
    return np.issubdtype(state.dtype, np.floating)


def _append_to_buffer(experience, buffer, sub_buffer, reset_flag):
    s, a, r, non_terminal, s_next, info, *_ = experience
    if s.shape != (84, 84):
        # Use only the last image to reduce memory
        s = s[-1]
        s_next = s_next[-1]
    if _is_float(s):
        s = _denormalize_state(s)
    if _is_float(s_next):
        s_next = _denormalize_state(s_next)
    assert s.shape == (84, 84)
    assert s.shape == s_next.shape

    experience = (s, a, r, non_terminal, s_next, info, reset_flag)
    removed = buffer.append_with_removed_item_check(experience)
    if removed is not None:
        sub_buffer.append(removed)
    return (0 == non_terminal)


def _getitem_from_buffer(index, buffer, sub_buffer, stacked_frames):
    (_, a, r, non_terminal, s_next, info, _) = buffer[index]
    states = np.zeros(shape=(stacked_frames, 84, 84), dtype=np.uint8)
    for i in range(0, stacked_frames):
        buffer_index = index - i
        if 0 <= buffer_index:
            (s, _, _, _, _, _, reset) = buffer[buffer_index]
        else:
            (s, _, _, _, _, _, reset) = sub_buffer[buffer_index]
        assert s.shape == (84, 84)
        tail_index = stacked_frames-i
        if reset:
            states[0:tail_index] = s
            break
        else:
            states[tail_index-1] = s
    s = _normalize_state(states)
    assert s.shape == (stacked_frames, 84, 84)

    s_next = np.expand_dims(s_next, axis=0)
    s_next = _normalize_state(s_next)
    if 1 < stacked_frames:
        s_next = np.concatenate((s[1:], s_next), axis=0)
    assert s.shape == s_next.shape
    return (s, a, r, non_terminal, s_next, info)
