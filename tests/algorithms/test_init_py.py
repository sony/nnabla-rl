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

from dataclasses import dataclass

import pytest

import nnabla_rl.algorithms as A
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig


@dataclass
class SampleConfig(AlgorithmConfig):
    pass


class SampleAlgorithm(Algorithm):
    def __init__(self):
        pass


class TestInitPy(object):
    def setup_method(self, method):
        pass

    def test_register_algorithm(self):
        assert not A.is_registered(SampleAlgorithm, SampleConfig)
        algorithm_num_before = len(A._ALGORITHMS)

        A.register_algorithm(SampleAlgorithm, SampleConfig)

        assert A.is_registered(SampleAlgorithm, SampleConfig)
        algorithm_num_after = len(A._ALGORITHMS)

        assert algorithm_num_before + 1 == algorithm_num_after

    def test_get_class_of(self):
        A.register_algorithm(SampleAlgorithm, SampleConfig)

        (algorithm_class, param_class) = A.get_class_of(SampleAlgorithm.__name__)
        assert algorithm_class == SampleAlgorithm
        assert param_class == SampleConfig

        with pytest.raises(KeyError):
            (algorithm_class, param_class) = A.get_class_of("UnknownClassName")


if __name__ == "__main__":
    pytest.main()
