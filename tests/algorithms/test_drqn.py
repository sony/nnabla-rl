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


class TestDRQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        drqn = A.DRQN(dummy_env)

        assert drqn.__name__ == 'DRQN'

    def test_continuous_action_env_unsupported(self):
        '''
        Check that error occurs when training on continuous action env
        '''

        dummy_env = E.DummyContinuous()
        config = A.DRQNConfig()
        with pytest.raises(Exception):
            A.DRQN(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        config = A.DRQNConfig()
        config.unroll_steps = 2
        config.start_timesteps = 10
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        drqn = A.DRQN(dummy_env, config=config)

        drqn.train_online(dummy_env, total_iterations=10)

    def test_run_online_training_multistep(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        config = A.DRQNConfig()
        config.num_steps = 2
        config.unroll_steps = 2
        config.start_timesteps = 5
        config.batch_size = 2
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        drqn = A.DRQN(dummy_env, config=config)

        drqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        unroll_steps = 2
        config = A.DRQNConfig()
        config.batch_size = batch_size
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        config.unroll_steps = unroll_steps

        drqn = A.DRQN(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size + unroll_steps + 1)
        # remove rnn_states info here because DRQN does not use rnn_state info to update recurrent layers
        # info is the last element of experience
        experiences = [(*experience[:-1], {}) for experience in experiences]
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        drqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        drqn = A.DRQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = drqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.DRQNConfig(unroll_steps=0)
        with pytest.raises(ValueError):
            A.DRQNConfig(unroll_steps=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        drqn = A.DRQN(dummy_env)

        drqn._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}

        latest_iteration_state = drqn.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
