import pytest

import nnabla as nn

import numpy as np

from nnabla_rl.algorithm import EnvironmentInfo
from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestBEAR(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        assert bear.__name__ == 'BEAR'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        params = A.BEARParam(start_timesteps=100)
        bear = A.BEAR(dummy_env, params=params)

        bear.train_online(dummy_env, total_iterations=10)

    def test_run_online_training_with_default_param(self):
        """
        Check that error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        with pytest.raises(ValueError):
            bear.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyContinuous()
        params = A.BEARParam(batch_size=batch_size)
        bear = A.BEAR(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        bear.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = bear.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_target_network_initialization(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        # Should be initialized to same parameters
        for q, target_q in zip(bear._q_ensembles, bear._target_q_ensembles):
            assert self._has_same_parameters(
                q.get_parameters(), target_q.get_parameters())

    def test_update_algorithm_params(self):
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        tau = 1.0
        gamma = 0.5
        learning_rate = 1e-5
        batch_size = 1000
        num_q_ensembles = 30
        num_mmd_actions = 5
        num_action_samples = 10
        warmup_iterations = 500
        start_timesteps = 2000
        param = {'tau': tau,
                 'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size,
                 'num_q_ensembles': num_q_ensembles,
                 'num_mmd_actions': num_mmd_actions,
                 'num_action_samples': num_action_samples,
                 'warmup_iterations': warmup_iterations,
                 'start_timesteps': start_timesteps}

        bear.update_algorithm_params(**param)

        assert bear._params.tau == tau
        assert bear._params.gamma == gamma
        assert bear._params.learning_rate == learning_rate
        assert bear._params.batch_size == batch_size
        assert bear._params.num_q_ensembles == num_q_ensembles
        assert bear._params.num_mmd_actions == num_mmd_actions
        assert bear._params.num_action_samples == num_action_samples
        assert bear._params.warmup_iterations == warmup_iterations
        assert bear._params.start_timesteps == start_timesteps

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.BEARParam(tau=1.1)
        with pytest.raises(ValueError):
            A.BEARParam(tau=-0.1)
        with pytest.raises(ValueError):
            A.BEARParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.BEARParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.BEARParam(num_q_ensembles=-100)
        with pytest.raises(ValueError):
            A.BEARParam(num_mmd_actions=-100)
        with pytest.raises(ValueError):
            A.BEARParam(num_action_samples=-100)
        with pytest.raises(ValueError):
            A.BEARParam(warmup_iterations=-100)
        with pytest.raises(ValueError):
            A.BEARParam(start_timesteps=-100)

    def test_compute_gaussian_mmd(self):
        def gaussian_kernel(x):
            return x**2
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                             [[2, 2, 2], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
        samples2 = np.array([[[0, 0, 0], [1, 1, 1]],
                             [[1, 2, 3], [1, 1, 1]]], dtype=np.float32)
        samples1_var = nn.Variable(samples1.shape)
        samples1_var.d = samples1
        samples2_var = nn.Variable(samples2.shape)
        samples2_var.d = samples2

        actual_mmd = bear._compute_gaussian_mmd(
            samples1=samples1_var, samples2=samples2_var, sigma=20.0)
        actual_mmd.forward()
        expected_mmd = self._compute_mmd(
            samples1, samples2, sigma=20.0, kernel=gaussian_kernel)

        assert actual_mmd.shape == (samples1.shape[0], 1)
        assert np.all(np.isclose(actual_mmd.d, expected_mmd))

    def test_compute_laplacian_mmd(self):
        def laplacian_kernel(x):
            return np.abs(x)
        dummy_env = E.DummyContinuous()
        bear = A.BEAR(dummy_env)

        samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                             [[2, 2, 2], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
        samples2 = np.array([[[0, 0, 0], [1, 1, 1]],
                             [[1, 2, 3], [1, 1, 1]]], dtype=np.float32)
        samples1_var = nn.Variable(samples1.shape)
        samples1_var.d = samples1
        samples2_var = nn.Variable(samples2.shape)
        samples2_var.d = samples2

        actual_mmd = bear._compute_laplacian_mmd(
            samples1=samples1_var, samples2=samples2_var, sigma=20.0)
        actual_mmd.forward()
        expected_mmd = self._compute_mmd(
            samples1, samples2, sigma=20.0, kernel=laplacian_kernel)

        assert actual_mmd.shape == (samples1.shape[0], 1)
        assert np.all(np.isclose(actual_mmd.d, expected_mmd))

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].data.data, params2[key].data.data):
                return False
        return True

    def _compute_mmd(self, samples1, samples2, sigma, kernel):
        diff_xx = self._compute_kernel_sum(samples1, samples1, sigma, kernel)
        diff_xy = self._compute_kernel_sum(samples1, samples2, sigma, kernel)
        diff_yy = self._compute_kernel_sum(samples2, samples2, sigma, kernel)
        n = samples1.shape[1]
        m = samples2.shape[1]
        mmd = (diff_xx / (n*n) -
               2.0 * diff_xy / (n*m) +
               diff_yy / (m*m))
        mmd = np.sqrt(mmd + 1e-6)
        return mmd

    def _compute_kernel_sum(self, a, b, sigma, kernel):
        sums = []
        for index in range(a.shape[0]):
            kernel_sum = 0.0
            for i in range(a.shape[1]):
                for j in range(b.shape[1]):
                    diff = 0.0
                    for k in range(a.shape[2]):
                        # print(f'samples[{i}] - samples[{j}]={samples1[i]-samples1[j]}')
                        diff += kernel(a[index][i][k]-b[index][j][k])
                    kernel_sum += np.exp(-diff/(2.0*sigma))
            sums.append(kernel_sum)
        return np.reshape(np.array(sums), newshape=(len(sums), 1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
