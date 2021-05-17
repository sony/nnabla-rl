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


class TestDummy(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscrete()
        dummy = A.Dummy(dummy_env)

        assert dummy.__name__ == 'Dummy'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        dummy.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        experience_num = 100
        fake_states = np.empty(shape=(experience_num, ) +
                               dummy_env.observation_space.shape)
        fake_actions = np.empty(shape=(experience_num, ) +
                                dummy_env.action_space.shape)
        fake_rewards = np.empty(shape=(experience_num, 1))
        fake_non_terminals = np.empty(shape=(experience_num, 1))
        fake_next_states = np.empty(shape=(experience_num, ) +
                                    dummy_env.observation_space.shape)
        fake_next_actions = np.empty(shape=(experience_num, ) +
                                     dummy_env.action_space.shape)

        fake_experiences = zip(fake_states, fake_actions, fake_rewards,
                               fake_non_terminals, fake_next_states, fake_next_actions)
        dummy.train_offline(fake_experiences, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        dummy = A.Dummy(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dummy.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape


if __name__ == "__main__":
    pytest.main()
