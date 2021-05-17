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

import nnabla as nn
import nnabla_rl.model_trainers as MT


class TestC51ValueDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


class TestQuantileDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_precompute_tau_hat(self):
        n_quantiles = 100

        expected = np.empty(shape=(n_quantiles,))
        prev_tau = 0.0

        for i in range(0, n_quantiles):
            tau = (i + 1) / n_quantiles
            expected[i] = (prev_tau + tau) / 2.0
            prev_tau = tau

        actual = MT.q_value.QRDQNQTrainer._precompute_tau_hat(n_quantiles)

        assert np.allclose(expected, actual)


class TestSquaredTDQFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


if __name__ == "__main__":
    pytest.main()
