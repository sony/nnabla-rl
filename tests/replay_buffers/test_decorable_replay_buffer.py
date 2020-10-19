import pytest
from unittest.mock import create_autospec

from nnabla_rl.replay_buffers.decorable_replay_buffer import DecorableReplayBuffer


class TestMDecorableReplayBuffer(object):
    def decor_fun(self, experience):
        pass

    def test_getitem(self):
        mock_decor_fun = create_autospec(
            self.decor_fun, return_value=(1, 2, 3, 4, 5))

        capacity = 10
        buffer = DecorableReplayBuffer(capacity=capacity,
                                       decor_fun=mock_decor_fun)

        append_num = 10
        for i in range(append_num):
            buffer.append(i)

        for _ in range(len(buffer)):
            experience = buffer[i]
            assert experience == (1, 2, 3, 4, 5)

        assert mock_decor_fun.call_count == len(buffer)


if __name__ == "__main__":
    pytest.main()
