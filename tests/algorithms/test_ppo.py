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

import pytest

import nnabla as nn
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E


class TestPPO(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        assert ppo.__name__ == 'PPO'

    def test_run_online_discrete_env_training(self):
        '''
        Check that no error occurs when calling online training (discrete env)
        '''

        dummy_env = E.DummyDiscreteImg()
        actor_timesteps = 10
        actor_num = 2
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_continuous_env_training(self):
        '''
        Check that no error occurs when calling online training (continuous env)
        '''

        dummy_env = E.DummyContinuous()
        actor_timesteps = 10
        actor_num = 2
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps * actor_num)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        with pytest.raises(ValueError):
            ppo.train_offline([], total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.PPOConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.PPOConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.PPOConfig(actor_num=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(actor_timesteps=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(total_timesteps=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuous()
        ppo = A.PPO(dummy_env)

        ppo._policy_trainer_state = {'pi_loss': 0.}
        ppo._v_function_trainer_state = {'v_loss': 1.}

        latest_iteration_state = ppo.latest_iteration_state
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'v_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['pi_loss'] == 0.
        assert latest_iteration_state['scalar']['v_loss'] == 1.


if __name__ == "__main__":
    pytest.main()
