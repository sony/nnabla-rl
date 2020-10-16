import pytest
from unittest.mock import create_autospec

import numpy as np

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
