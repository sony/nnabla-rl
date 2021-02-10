# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import numpy as np

import nnabla_rl.environments as E
import nnabla_rl.model_trainers as MT
from nnabla_rl.environments.environment_info import EnvironmentInfo


class TestC51ValueDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


class TestQuantileDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_precompute_tau_hat(self):
        dummy_env = E.DummyDiscreteImg()
        env_info = EnvironmentInfo.from_env(dummy_env)
        n_quantiles = 100

        config = MT.q_value_trainers.QRDQNQuantileDistributionFunctionTrainerConfig(num_quantiles=n_quantiles)
        trainer = MT.q_value_trainers.QRDQNQuantileDistributionFunctionTrainer(env_info, config=config)

        expected = np.empty(shape=(n_quantiles,))
        prev_tau = 0.0

        for i in range(0, n_quantiles):
            tau = (i + 1) / n_quantiles
            expected[i] = (prev_tau + tau) / 2.0
            prev_tau = tau

        actual = trainer._precompute_tau_hat(n_quantiles)

        assert np.allclose(expected, actual)


class TestSquaredTDQFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


if __name__ == "__main__":
    pytest.main()
