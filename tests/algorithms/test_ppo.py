import pytest

import nnabla as nn

import numpy as np

import nnabla_rl as rl
from nnabla_rl.algorithm import EnvironmentInfo
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestPPO(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        assert ppo.__name__ == 'PPO'

    def test_run_online_discrete_env_training(self):
        """
        Check that no error occurs when calling online training (discrete env)
        """

        dummy_env = E.DummyDiscreteImg()
        params = A.PPOParam(batch_size=5, actor_timesteps=10, actor_num=2)
        ppo = A.PPO(dummy_env, params=params)

        ppo.train_online(dummy_env, total_iterations=2)

    def test_run_online_continuous_env_training(self):
        """
        Check that no error occurs when calling online training (continuous env)
        """

        dummy_env = E.DummyContinuous()
        params = A.PPOParam(batch_size=5, actor_timesteps=10, actor_num=2)
        ppo = A.PPO(dummy_env, params=params)

        ppo.train_online(dummy_env, total_iterations=2)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        with pytest.raises(ValueError):
            ppo.train_offline([], total_iterations=10)

    def test_update_algorithm_params(self):
        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        gamma = 0.5
        learning_rate = 1e-5
        batch_size = 1000
        param = {'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size}

        ppo.update_algorithm_params(**param)

        assert ppo._params.gamma == gamma
        assert ppo._params.learning_rate == learning_rate
        assert ppo._params.batch_size == batch_size

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.PPOParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.PPOParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.PPOParam(actor_num=-1)
        with pytest.raises(ValueError):
            A.PPOParam(batch_size=-1)
        with pytest.raises(ValueError):
            A.PPOParam(actor_timesteps=-1)
        with pytest.raises(ValueError):
            A.PPOParam(total_timesteps=-1)


if __name__ == "__main__":
    pytest.main()
