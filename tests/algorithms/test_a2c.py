# Copyright 2021 Sony Corporation.
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

import pytest

import nnabla as nn
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E


class TestA2C(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        a2c = A.A2C(dummy_env)

        assert a2c.__name__ == 'A2C'

    def test_continuous_env_unsupported(self):
        '''
        Check that error occurs when training on continuous env
        '''

        dummy_env = E.DummyContinuous()
        config = A.A2CConfig()
        with pytest.raises(Exception):
            A.A2C(dummy_env, config=config)

    def test_run_online_discrete_env_training(self):
        '''
        Check that no error occurs when calling online training (discrete env)
        '''

        dummy_env = E.DummyDiscreteImg()
        n_steps = 4
        actor_num = 2
        config = A.A2CConfig(n_steps=n_steps, actor_num=actor_num)
        a2c = A.A2C(dummy_env, config=config)

        a2c.train_online(dummy_env, total_iterations=n_steps*actor_num)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyDiscreteImg()
        a2c = A.A2C(dummy_env)

        with pytest.raises(ValueError):
            a2c.train_offline([], total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.A2CConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.A2CConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.A2CConfig(decay=1.1)
        with pytest.raises(ValueError):
            A.A2CConfig(decay=-0.1)
        with pytest.raises(ValueError):
            A.A2CConfig(n_steps=-1)
        with pytest.raises(ValueError):
            A.A2CConfig(actor_num=-1)
        with pytest.raises(ValueError):
            A.A2CConfig(learning_rate=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        a2c = A.A2C(dummy_env)

        a2c._policy_trainer_state = {'pi_loss': 0.}
        a2c._v_function_trainer_state = {'v_loss': 1.}

        latest_iteration_state = a2c.latest_iteration_state
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'v_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['pi_loss'] == 0.
        assert latest_iteration_state['scalar']['v_loss'] == 1.


if __name__ == "__main__":
    pytest.main()
