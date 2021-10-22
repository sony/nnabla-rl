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

from nnabla_rl.environments.dummy import DummyAtariEnv
from nnabla_rl.environments.wrappers.atari import MaxAndSkipEnv, NoopResetEnv
from nnabla_rl.replay_buffers.memory_efficient_atari_buffer import (MemoryEfficientAtariBuffer,
                                                                    ProportionalPrioritizedAtariBuffer,
                                                                    RankBasedPrioritizedAtariBuffer)
from nnabla_rl.utils.reproductions import build_atari_env


class TestMemoryEfficientAtariBuffer(object):
    def test_append_float(self):
        experience = _generate_atari_experience_mock()[0]

        capacity = 10
        buffer = MemoryEfficientAtariBuffer(capacity=capacity)
        buffer.append(experience)

        s, _, _, _, s_next, *_ = buffer._buffer[len(buffer._buffer)-1]
        assert s.dtype == np.uint8
        assert s_next.dtype == np.uint8
        assert np.alltrue(
            (experience[0][-1] * 255.0).astype(np.uint8) == s)
        assert np.alltrue(
            (experience[4][-1] * 255.0).astype(np.uint8) == s_next)

    def test_unstacked_frame(self):
        experiences = _generate_atari_experience_mock(num_mocks=10, frame_stack=False)

        capacity = 10
        buffer = MemoryEfficientAtariBuffer(capacity=capacity, stacked_frames=1)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.shape[0] == 1
            assert s_next.shape[0] == 1
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_getitem(self):
        experiences = _generate_atari_experience_mock(num_mocks=10)

        capacity = 10
        buffer = MemoryEfficientAtariBuffer(capacity=capacity)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_full_buffer_getitem(self):
        capacity = 10
        experiences = _generate_atari_experience_mock(
            num_mocks=(capacity + 5))
        buffer = MemoryEfficientAtariBuffer(capacity=capacity)

        for i in range(capacity):
            buffer.append(experiences[i])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

        for i in range(5):
            buffer.append(experiences[i + capacity])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i + 5]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_buffer_len(self):
        capacity = 10
        buffer = MemoryEfficientAtariBuffer(capacity=capacity)
        for _ in range(10):
            experience = _generate_atari_experience_mock()[0]
            buffer.append(experience)

        assert len(buffer) == 10


class TestProportionalPrioritizedAtariBuffer(object):
    def test_append_float(self):
        experience = _generate_atari_experience_mock()[0]

        capacity = 10
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity)
        buffer.append(experience)

        s, _, _, _, s_next, *_ = buffer._buffer[len(buffer._buffer)-1]
        assert s.dtype == np.uint8
        assert s_next.dtype == np.uint8
        assert np.alltrue(
            (experience[0][-1] * 255.0).astype(np.uint8) == s)
        assert np.alltrue(
            (experience[4][-1] * 255.0).astype(np.uint8) == s_next)

    def test_unstacked_frame(self):
        experiences = _generate_atari_experience_mock(num_mocks=10, frame_stack=False)

        capacity = 10
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity, stacked_frames=1)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.shape[0] == 1
            assert s_next.shape[0] == 1
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_getitem(self):
        experiences = _generate_atari_experience_mock(num_mocks=10)

        capacity = 10
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_full_buffer_getitem(self):
        capacity = 10
        experiences = _generate_atari_experience_mock(
            num_mocks=(capacity + 5))
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity)

        for i in range(capacity):
            buffer.append(experiences[i])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

        for i in range(5):
            buffer.append(experiences[i + capacity])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i + 5]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_buffer_len(self):
        capacity = 10
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity)
        for _ in range(10):
            experience = _generate_atari_experience_mock()[0]
            buffer.append(experience)

        assert len(buffer) == 10

    def test_sample_without_update(self):
        beta = 0.5
        capacity = 10
        buffer = ProportionalPrioritizedAtariBuffer(capacity=capacity, beta=beta)
        for _ in range(5):
            experience = _generate_atari_experience_mock()[0]
            buffer.append(experience)

        indices = [1, 3, 2]
        _, weights = buffer.sample_indices(indices)

        # update the priority and check that following sampling succeeds
        errors = np.random.sample([len(weights), 1])
        buffer.update_priorities(errors)

        _, _ = buffer.sample_indices(indices)

        # sample without priority update
        with pytest.raises(RuntimeError):
            buffer.sample_indices(indices)


class TestRankBasedPrioritizedAtariBuffer(object):
    def test_append_float(self):
        experience = _generate_atari_experience_mock()[0]

        capacity = 10
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity)
        buffer.append(experience)

        s, _, _, _, s_next, *_ = buffer._buffer[len(buffer._buffer)-1]
        assert s.dtype == np.uint8
        assert s_next.dtype == np.uint8
        assert np.alltrue(
            (experience[0][-1] * 255.0).astype(np.uint8) == s)
        assert np.alltrue(
            (experience[4][-1] * 255.0).astype(np.uint8) == s_next)

    def test_unstacked_frame(self):
        experiences = _generate_atari_experience_mock(num_mocks=10, frame_stack=False)

        capacity = 10
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity, stacked_frames=1)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.shape[0] == 1
            assert s_next.shape[0] == 1
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_getitem(self):
        experiences = _generate_atari_experience_mock(num_mocks=10)

        capacity = 10
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity)

        for experience in experiences:
            buffer.append(experience)

        for i, experience in enumerate(experiences):
            s, _, _, _, s_next, *_ = buffer.__getitem__(i)
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_full_buffer_getitem(self):
        capacity = 10
        experiences = _generate_atari_experience_mock(
            num_mocks=(capacity + 5))
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity)

        for i in range(capacity):
            buffer.append(experiences[i])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

        for i in range(5):
            buffer.append(experiences[i + capacity])
        assert len(buffer) == capacity

        for i in range(capacity):
            s, _, _, _, s_next, *_ = buffer[i]
            experience = experiences[i + 5]
            assert s.dtype == np.float32
            assert s_next.dtype == np.float32
            assert np.allclose(experience[0], s, atol=1e-2)
            assert np.allclose(experience[4], s_next, atol=1e-2)

    def test_buffer_len(self):
        capacity = 10
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity)
        for _ in range(10):
            experience = _generate_atari_experience_mock()[0]
            buffer.append(experience)

        assert len(buffer) == 10

    def test_sample_without_update(self):
        beta = 0.5
        capacity = 10
        buffer = RankBasedPrioritizedAtariBuffer(capacity=capacity, beta=beta)
        for _ in range(5):
            experience = _generate_atari_experience_mock()[0]
            buffer.append(experience)

        indices = [1, 3, 2]
        _, weights = buffer.sample_indices(indices)

        # update the priority and check that following sampling succeeds
        errors = np.random.sample([len(weights), 1])
        buffer.update_priorities(errors)

        _, _ = buffer.sample_indices(indices)

        # sample without priority update
        with pytest.raises(RuntimeError):
            buffer.sample_indices(indices)


def _generate_atari_experience_mock(low=0.0, high=1.0, num_mocks=1, frame_stack=True):
    env = DummyAtariEnv()
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = build_atari_env(env, test=True, print_info=False, frame_stack=frame_stack)
    experiences = []
    state = env.reset()
    for _ in range(num_mocks):
        action = env.action_space.sample()
        s_next, reward, done, info = env.step(action)
        experience = (state, action, reward, 1.0 - done, s_next, info)
        experiences.append(experience)
        if done:
            state = env.reset()
        else:
            state = s_next
    return experiences


if __name__ == "__main__":
    pytest.main()
