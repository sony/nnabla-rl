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

import nnabla_rl.random as random


class TestRandom(object):
    def test_random_seed(self):
        random.seed(0)
        samples1 = random.drng.choice(10, size=5)

        random.seed(0)
        samples2 = random.drng.choice(10, size=5)

        assert len(samples1) == 5
        assert len(samples1) == len(samples2)
        assert np.allclose(samples1, samples2)

        random.seed(100)
        samples3 = random.drng.choice(10, size=5)

        assert len(samples3) == 5
        assert not np.allclose(samples1, samples3)


if __name__ == "__main__":
    pytest.main()
