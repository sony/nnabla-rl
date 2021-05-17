# Copyright 2021 Sony Corporation.
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

from nnabla_rl.replay_buffers.replacement_sampling_replay_buffer import ReplacementSamplingReplayBuffer


class TestReplacementSamplingReplayBuffer():
    def test_sample_from_insufficient_size_buffer(self):
        buffer = self._generate_buffer_with_experiences(experience_num=10)
        samples, _ = buffer.sample(num_samples=100)
        assert len(samples) == 100

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
        buffer = ReplacementSamplingReplayBuffer()
        for _ in range(experience_num):
            experience = self._generate_experience_mock()
            buffer.append(experience)
        return buffer
