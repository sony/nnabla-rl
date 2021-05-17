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
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.replay_buffer import ReplayBuffer


class TestSAC(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        assert sac.__name__ == 'SAC'

    def test_discrete_env_unsupported(self):
        '''
        Check that error occurs when training on discrete env
        '''

        dummy_env = E.DummyDiscrete()
        with pytest.raises(Exception):
            A.SAC(dummy_env)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        sac.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        batch_size = 5
        dummy_env = E.DummyContinuous()
        config = A.SACConfig(batch_size=batch_size)
        sac = A.SAC(dummy_env, config=config)

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

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.SACConfig(tau=1.1)
        with pytest.raises(ValueError):
            A.SACConfig(tau=-0.1)
        with pytest.raises(ValueError):
            A.SACConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.SACConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.SACConfig(start_timesteps=-100)
        with pytest.raises(ValueError):
            A.SACConfig(environment_steps=-100)
        with pytest.raises(ValueError):
            A.SACConfig(gradient_steps=-100)
        with pytest.raises(ValueError):
            A.SACConfig(initial_temperature=-100)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuous()
        sac = A.SAC(dummy_env)

        sac._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}
        sac._policy_trainer_state = {'pi_loss': 1.}

        latest_iteration_state = sac.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert latest_iteration_state['scalar']['pi_loss'] == 1.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
