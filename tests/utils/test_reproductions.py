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

import nnabla.functions as NF
import nnabla.initializer as NI
import nnabla_rl.utils.reproductions as reproductions


class TestReproductions():
    def test_set_global_seed(self):
        seed = 0

        reproductions.set_global_seed(seed)
        random_integer1 = np.random.randint(low=10)
        initializer1 = NI.UniformInitializer()
        init_param1 = initializer1(shape=(10, 10))
        random_variable1 = NF.randn(shape=(10, 10))
        random_variable1.forward()

        reproductions.set_global_seed(seed)
        random_integer2 = np.random.randint(low=10)
        initializer2 = NI.UniformInitializer()
        init_param2 = initializer2(shape=(10, 10))
        random_variable2 = NF.randn(shape=(10, 10))
        random_variable2.forward()

        assert random_integer1 == random_integer2
        assert np.allclose(init_param1, init_param2)
        assert np.allclose(random_variable1.d, random_variable2.d)

        # Check that different random seed gives different result
        another_seed = 10
        reproductions.set_global_seed(another_seed)

        random_integer3 = np.random.randint(low=10)
        initializer3 = NI.UniformInitializer()
        init_param3 = initializer3(shape=(10, 10))
        random_variable3 = NF.randn(shape=(10, 10))
        random_variable3.forward()

        assert random_integer1 != random_integer3
        assert not np.allclose(init_param1, init_param3)
        assert not np.allclose(random_variable1.d, random_variable3.d)


if __name__ == '__main__':
    pytest.main()
