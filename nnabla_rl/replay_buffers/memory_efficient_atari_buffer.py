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

from collections import deque

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.data import RingBuffer


class MemoryEfficientAtariBuffer(ReplayBuffer):
    '''Buffer designed to compactly save experiences of Atari environments used in DQN.

    DQN (and other training algorithms) requires large replay buffer when training on Atari games.
    If you naively save the experiences, you'll need more than 100GB to save them (assuming 1M experiences).
    Which usually does not fit in the machine's memory (unless you have money:).
    This replay buffer reduces the size of experience by casting the images to uint8 and
    removing old frames concatenated to the observation.
    By using this buffer, you can hold 1M experiences using only 20GB(approx.) of memory.

    Note that this class is designed only for DQN style training on atari environment.
    (i.e. State consists of 4 concatenated grayscaled frames and its values are normalized between 0 and 1)
    '''

    def __init__(self, capacity):
        super(MemoryEfficientAtariBuffer, self).__init__(capacity=capacity)
        self._scale = 255.0
        self._reset = True
        self._buffer = RingBuffer(maxlen=capacity)
        self._sub_buffer = deque(maxlen=3)

    def append(self, experience):
        s, a, r, non_terminal, s_next, *_ = experience
        if not self._is_float(s):
            raise ValueError('dtype {} is not supported'.format(s.dtype))
        if not self._is_float(s_next):
            raise ValueError('dtype {} is not supported'.format(s_next.dtype))

        # cast to uint8 and use only the last image to reduce memory
        s = self._denormalize_state(s[-1])
        s_next = self._denormalize_state(s_next[-1])
        s = np.array(s, copy=True, dtype=np.uint8)
        s_next = np.array(s_next, copy=True, dtype=np.uint8)
        assert s.shape == (84, 84)
        assert s.shape == s_next.shape
        experience = (s, a, r, non_terminal, s_next, self._reset)
        removed = self._buffer.append_with_removed_item_check(experience)
        if removed is not None:
            self._sub_buffer.append(removed)
        self._reset = (0 == non_terminal)

    def append_all(self, experiences):
        for experience in experiences:
            self.append(experience)

    def __getitem__(self, index):
        (_, a, r, non_terminal, s_next, _) = self._buffer[index]
        states = np.empty(shape=(4, 84, 84), dtype=np.uint8)
        for i in range(0, 4):
            buffer_index = index - i
            if 0 <= buffer_index:
                (s, _, _, _, _, reset) = self._buffer[buffer_index]
            else:
                (s, _, _, _, _, reset) = self._sub_buffer[buffer_index]
            assert s.shape == (84, 84)
            tail_index = 4-i
            if reset:
                states[0:tail_index] = s
                break
            else:
                states[tail_index-1] = s
        s = self._normalize_state(states)
        assert s.shape == (4, 84, 84)

        s_next = np.expand_dims(s_next, axis=0)
        s_next = self._normalize_state(s_next)
        s_next = np.concatenate((s[1:], s_next), axis=0)
        assert s.shape == s_next.shape
        return (s, a, r, non_terminal, s_next)

    def _denormalize_state(self, state):
        return (state * self._scale).astype(np.uint8)

    def _normalize_state(self, state):
        return state.astype(np.float32) / self._scale

    def _is_float(self, state):
        return np.issubdtype(state.dtype, np.floating)
