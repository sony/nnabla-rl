import pytest

import nnabla as nn
import nnabla.functions as F

import numpy as np

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A
from nnabla_rl.algorithms.sac import AdjustableTemperature


class TestAdjustableTemperature(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_initial_temperature(self):
        initial_value = 5.0
        temperature = AdjustableTemperature(
            scope_name='test', initial_value=initial_value)
        actual_value = temperature()
        actual_value.forward(clear_no_need_grad=True)

        assert actual_value.data.data == initial_value

        # Create tempearture with random initial value
        nn.clear_parameters()
        temperature = AdjustableTemperature(scope_name='test')
        actual_value = temperature()
        actual_value.forward(clear_no_need_grad=True)

        # No error occurs -> pass

    def test_temperature_is_adjustable(self):
        initial_value = 5.0
        temperature = AdjustableTemperature(
            scope_name='test', initial_value=initial_value)
        solver = nn.solvers.Adam(alpha=1.0)
        solver.set_parameters(temperature.get_parameters())

        value = temperature()

        loss = 0.5 * F.mean(value ** 2)
        loss.forward()

        solver.zero_grad()
        loss.backward()

        solver.update()

        updated_value = temperature()
        updated_value.forward(clear_no_need_grad=True)

        new_value = updated_value.data.data
        assert not np.isclose(new_value, initial_value)
        assert new_value < initial_value


class TestSAC(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        assert sac.__name__ == 'SAC'

    def test_run_online_training(self):
        """
        Check that no error occurs when calling online training
        """

        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        sac.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """
        Check that no error occurs when calling offline training
        """

        batch_size = 5
        dummy_env = E.DummyContinuous()
        params = A.SACParam(batch_size=batch_size)
        sac = A.SAC(dummy_env, params=params)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        sac.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = sac.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_target_network_initialization(self):
        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        # Should be initialized to same parameters
        assert self._has_same_parameters(
            sac._q1.get_parameters(), sac._target_q1.get_parameters())
        assert self._has_same_parameters(
            sac._q2.get_parameters(), sac._target_q2.get_parameters())

    def test_update_algorithm_params(self):
        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        tau = 1.0
        gamma = 0.5
        learning_rate = 1e-5
        batch_size = 1000
        start_timesteps = 10
        replay_buffer_size = 100
        environment_steps: int = 5
        gradient_steps: int = 10
        target_entropy: float = -100
        fix_temperature: bool = True

        param = {'tau': tau,
                 'gamma': gamma,
                 'learning_rate': learning_rate,
                 'batch_size': batch_size,
                 'start_timesteps': start_timesteps,
                 'replay_buffer_size': replay_buffer_size,
                 'environment_steps': environment_steps,
                 'gradient_steps': gradient_steps,
                 'target_entropy': target_entropy,
                 'fix_temperature': fix_temperature}

        sac.update_algorithm_params(**param)

        assert sac._params.tau == tau
        assert sac._params.gamma == gamma
        assert sac._params.learning_rate == learning_rate
        assert sac._params.batch_size == batch_size
        assert sac._params.start_timesteps == start_timesteps
        assert sac._params.replay_buffer_size == replay_buffer_size
        assert sac._params.gradient_steps == gradient_steps
        assert sac._params.environment_steps == environment_steps
        assert sac._params.target_entropy == target_entropy
        assert sac._params.fix_temperature == fix_temperature

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.SACParam(tau=1.1)
        with pytest.raises(ValueError):
            A.SACParam(tau=-0.1)
        with pytest.raises(ValueError):
            A.SACParam(gamma=1.1)
        with pytest.raises(ValueError):
            A.SACParam(gamma=-0.1)
        with pytest.raises(ValueError):
            A.SACParam(start_timesteps=-100)
        with pytest.raises(ValueError):
            A.SACParam(environment_steps=-100)
        with pytest.raises(ValueError):
            A.SACParam(gradient_steps=-100)
        with pytest.raises(ValueError):
            A.SACParam(initial_temperature=-100)

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].data.data, params2[key].data.data):
                return False
        return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
