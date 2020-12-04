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

        params = MT.q_value_trainers.QRDQNQuantileDistributionFunctionTrainerParam(num_quantiles=n_quantiles)
        trainer = MT.q_value_trainers.QRDQNQuantileDistributionFunctionTrainer(env_info, params=params)

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
