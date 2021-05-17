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


class TestIQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        assert iqn.__name__ == 'IQN'

    def test_continuous_env_unsupported(self):
        '''
        Check that error occurs when training on continuous env
        '''

        dummy_env = E.DummyContinuous()
        config = A.IQNConfig()
        with pytest.raises(Exception):
            A.IQN(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.IQNConfig()
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        iqn = A.IQN(dummy_env, config=config)

        iqn.train_online(dummy_env, total_iterations=5)

    def test_run_offline_training(self):
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        config = A.IQNConfig()
        config.batch_size = batch_size
        config.learner_update_frequency = 1
        config.target_update_frequency = 1

        iqn = A.IQN(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        iqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = iqn.compute_eval_action(state)

        assert action.shape == (1, )

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.IQNConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.IQNConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.IQNConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(learner_update_frequency=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(max_explore_steps=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(learning_rate=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(initial_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(final_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(test_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(K=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(N=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(N_prime=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(kappa=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(embedding_dim=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        iqn._quantile_function_trainer_state = {'q_loss': 0.}

        latest_iteration_state = iqn.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['q_loss'] == 0.


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
