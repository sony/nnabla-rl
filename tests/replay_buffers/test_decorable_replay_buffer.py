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

from unittest.mock import create_autospec

import pytest

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
