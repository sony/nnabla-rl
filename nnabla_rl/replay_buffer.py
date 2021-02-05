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

from collections import deque

import random
import numpy as np

from nnabla_rl.utils.data import RingBuffer


class ReplayBuffer(object):
    def __init__(self, capacity=None):
        assert capacity is None or capacity >= 0
        self._capacity = capacity

        if capacity is None:
            self._buffer = deque(maxlen=capacity)
        else:
            self._buffer = RingBuffer(maxlen=capacity)

    def __getitem__(self, item):
        return self._buffer[item]

    @property
    def capacity(self):
        '''
        Capacity (max length) of this replay buffer otherwise None
        '''
        return self._capacity

    def append(self, experience):
        '''
        Add new experience to the replay buffer.

        Args:
            experience (array-like): Experience includes trainsitions,
                such as state, action, reward, the iteration of environment has done or not.
                Please see to get more information in [Replay buffer documents](replay_buffer.md)

        Notes:
            If the replay buffer size is full, the oldest (head of the buffer) experience will be dropped off
            and the given experince will be added to the tail of the buffer.
        '''
        self._buffer.append(experience)

    def append_all(self, experiences):
        '''
        Add list of experiences to the replay buffer.

        Args:
            experiences (array-like): List of experiences to insert to the buffer
                Please see to get more information in [Replay buffer documents](replay_buffer.md)

        Notes:
            If the replay buffer size is full, the oldest (head of the buffer) experience will be dropped off
            and the given experince will be added to the tail of the buffer.
        '''
        for experience in experiences:
            self._buffer.append(experience)

    def sample(self, num_samples=1):
        '''
        Randomly sample num_samples experiences from the replay buffer.

        Args:
            num_samples (int): Number of samples to sample from the replay buffer.

        Returns:
            experiences (array-like): Random num_samples of experiences.
            info (dict): dictionary of information about experiences.

        Raises:
            ValueError: If num_samples is greater than current buffer size

        Notes
        ----
        Sampling strategy depends on the undelying implementation.
        '''
        buffer_length = len(self)
        if num_samples > buffer_length:
            raise ValueError(
                'num_samples: {} is greater than the size of buffer: {}'.format(num_samples, buffer_length))
        indices = self._random_indices(num_samples=num_samples)
        return self.sample_indices(indices)

    def sample_indices(self, indices):
        '''
        Sample experiences for given indices from the replay buffer.

        Args:
            indices (array-like): list of array index to sample the data

        Returns:
            experiences (array-like): Sample of experiences for given indices.

        Raises:
            ValueError: If indices are empty

        '''
        if len(indices) == 0:
            raise ValueError('Indices are empty')
        weights = np.ones([len(indices), 1])
        return [self.__getitem__(index) for index in indices], dict(weights=weights)

    def update_priorities(self, errors):
        pass

    def __len__(self):
        return len(self._buffer)

    def _random_indices(self, num_samples):
        buffer_length = len(self)
        return random.sample(range(buffer_length), k=num_samples)
