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
