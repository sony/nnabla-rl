import pytest

from nnabla_rl.utils.data import RingBuffer


class TestRingBuffer(object):
    def test_append(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            assert len(buffer) == i
            buffer.append(i)
        assert len(buffer) == maxlen

        for i in range(maxlen):
            assert len(buffer) == maxlen
            buffer.append(i)
        assert len(buffer) == maxlen

    def test_getitem(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            buffer.append(i)
            assert i == buffer[i]

        for i in range(maxlen):
            buffer.append(i + maxlen)
            assert i + 1 == buffer[0]
            assert i + maxlen == buffer[maxlen - 1]

    def test_buffer_len(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            assert len(buffer) == i
            buffer.append(i)
        assert len(buffer) == maxlen

        for i in range(maxlen):
            assert len(buffer) == maxlen
            buffer.append(i)
        assert len(buffer) == maxlen


if __name__ == "__main__":
    pytest.main()
