import pytest

import nnabla as nn

import numpy as np

import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestTRPO():
    def setup_method(self):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        assert trpo.__name__ == 'TRPO'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """
        dummy_env = E.DummyContinuous()
        dummy_env = EpisodicEnv(dummy_env, min_episode_length=3)

        params = A.TRPOParam(num_steps_per_iteration=5,
                             gpu_batch_size=5,
                             pi_batch_size=5,
                             vf_batch_size=2,
                             sigma_kl_divergence_constraint=10.0,
                             maximum_backtrack_numbers=50)
        trpo = A.TRPO(dummy_env, params=params)

        trpo.train_online(dummy_env, total_iterations=5)

    def test_run_offline_training(self):
        """
        Check that raising error when calling offline training
        """
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        with pytest.raises(NotImplementedError):
            trpo.train_offline([], total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        trpo = A.TRPO(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = trpo.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.TRPOParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(num_steps_per_iteration=-1)
        with pytest.raises(ValueError):
            A.TRPOParam(sigma_kl_divergence_constraint=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(maximum_backtrack_numbers=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(conjugate_gradient_damping=-0.1)
        with pytest.raises(ValueError):
            A.TRPOParam(conjugate_gradient_iterations=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_epochs=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_batch_size=-5)
        with pytest.raises(ValueError):
            A.TRPOParam(vf_learning_rate=-0.5)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv
    pytest.main()
else:
    from .testing_utils import EpisodicEnv
