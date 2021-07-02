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


class TestRainbow(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        config = A.RainbowConfig()
        config.replay_buffer_size = 10
        rainbow = A.Rainbow(dummy_env, config=config)

        assert rainbow.__name__ == 'Rainbow'

    def test_continuous_action_env_unsupported(self):
        '''
        Check that error occurs when training on continuous action env
        '''

        dummy_env = E.DummyContinuous()
        config = A.RainbowConfig()
        config.replay_buffer_size = 3
        with pytest.raises(Exception):
            A.Rainbow(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        config = A.RainbowConfig()
        config.start_timesteps = 5
        config.batch_size = 1
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        config.num_steps = 1
        config.replay_buffer_size = 3
        rainbow = A.Rainbow(dummy_env, config=config)
        rainbow.train_online(dummy_env, total_iterations=6)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        batch_size = 5
        dummy_env = E.DummyDiscreteImg()
        config = A.RainbowConfig(batch_size=batch_size)
        config.num_steps = 1
        config.batch_size = 1
        config.replay_buffer_size = 3
        rainbow = A.Rainbow(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        rainbow.train_offline(buffer, total_iterations=6)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        config = A.RainbowConfig()
        config.replay_buffer_size = 3
        rainbow = A.Rainbow(dummy_env, config=config)

        state = dummy_env.reset()
        state = np.float32(state)
        action = rainbow.compute_eval_action(state)

        assert action.shape == (1, )

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.RainbowConfig()
        config.replay_buffer_size = 3
        rainbow = A.Rainbow(dummy_env, config=config)

        rainbow._model_trainer_state = {'cross_entropy_loss': 0., 'td_errors': np.array([0., 1.])}

        latest_iteration_state = rainbow.latest_iteration_state
        assert 'cross_entropy_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['cross_entropy_loss'] == 0.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
