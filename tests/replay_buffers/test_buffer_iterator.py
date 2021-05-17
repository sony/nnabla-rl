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

import nnabla_rl as rl
from nnabla_rl.replay_buffers.buffer_iterator import BufferIterator


class TestBufferIterator(object):
    def test_buffer_iterator_shuffle_no_repeat(self):
        buffer_size = 100
        dummy_examples = np.arange(buffer_size)
        buffer = rl.replay_buffer.ReplayBuffer()
        buffer.append_all(dummy_examples)

        batch_size = 30
        iterator = BufferIterator(
            buffer=buffer, batch_size=batch_size, shuffle=True, repeat=False)

        for _ in range(buffer_size // batch_size):
            batch, *_ = iterator.next()
            assert len(batch) == batch_size
            assert not iterator.is_new_epoch()
        batch, *_ = iterator.next()
        assert len(batch) == (buffer_size % batch_size)
        assert iterator.is_new_epoch()

        with pytest.raises(StopIteration):
            iterator.next()

    def test_buffer_iterator_shuffle_with_repeat(self):
        buffer_size = 100
        dummy_examples = np.arange(buffer_size)
        buffer = rl.replay_buffer.ReplayBuffer()
        buffer.append_all(dummy_examples)

        batch_size = 30
        iterator = BufferIterator(
            buffer=buffer, batch_size=batch_size, shuffle=True, repeat=True)

        for _ in range(buffer_size // batch_size):
            batch, *_ = iterator.next()
            assert len(batch) == batch_size
            assert not iterator.is_new_epoch()
        batch, *_ = iterator.next()
        assert len(batch) == batch_size
        assert iterator.is_new_epoch()

        with pytest.raises(StopIteration):
            iterator.next()

        batch, *_ = iterator.next()
        assert len(batch) == batch_size
        assert not iterator.is_new_epoch()

    def test_buffer_iterator_is_iterable(self):
        buffer_size = 100
        dummy_examples = np.arange(buffer_size)
        buffer = rl.replay_buffer.ReplayBuffer()
        buffer.append_all(dummy_examples)

        batch_size = 30
        iterator = BufferIterator(
            buffer=buffer, batch_size=batch_size, shuffle=True, repeat=True)

        for experience, *_ in iterator:
            assert len(experience) == batch_size


if __name__ == "__main__":
    pytest.main()
