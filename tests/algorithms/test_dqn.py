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

from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl.algorithms as A


class TestDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        assert dqn.__name__ == 'DQN'

    def test_continuous_env_unsupported(self):
        '''
        Check that error occurs when training on continuous env
        '''

        dummy_env = E.DummyContinuous()
        config = A.DQNConfig()
        with pytest.raises(Exception):
            A.DQN(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        config = A.DQNConfig()
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        dqn = A.DQN(dummy_env, config=config)

        dqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        config = A.DQNConfig()
        config.batch_size = batch_size
        config.learner_update_frequency = 1
        config.target_update_frequency = 1

        dqn = A.DQN(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        dqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.DQNConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.DQNConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.DQNConfig(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.DQNConfig(learner_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.DQNConfig(target_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.DQNConfig(start_timesteps=-1000)
        with pytest.raises(ValueError):
            A.DQNConfig(replay_buffer_size=-1000)
        with pytest.raises(ValueError):
            A.DQNConfig(initial_epsilon=1.5)
        with pytest.raises(ValueError):
            A.DQNConfig(final_epsilon=1.1)
        with pytest.raises(ValueError):
            A.DQNConfig(test_epsilon=-1000)
        with pytest.raises(ValueError):
            A.DQNConfig(max_explore_steps=-100)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import generate_dummy_experiences
