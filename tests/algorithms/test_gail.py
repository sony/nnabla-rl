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

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.replay_buffer import ReplayBuffer


class TestGAIL():
    def setup_method(self):
        nn.clear_parameters()

    def _create_dummy_buffer(self, env, batch_size=5):
        experiences = generate_dummy_experiences(env, batch_size)
        dummy_buffer = ReplayBuffer()
        dummy_buffer.append_all(experiences)
        return dummy_buffer

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        dummy_buffer = self._create_dummy_buffer(dummy_env)
        gail = A.GAIL(dummy_env, dummy_buffer)

        assert gail.__name__ == 'GAIL'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyContinuous()
        dummy_env = EpisodicEnv(dummy_env, min_episode_length=3)
        dummy_buffer = self._create_dummy_buffer(dummy_env, batch_size=15)

        config = A.GAILConfig(num_steps_per_iteration=5,
                              pi_batch_size=5,
                              vf_batch_size=2,
                              discriminator_batch_size=2,
                              sigma_kl_divergence_constraint=10.0,
                              maximum_backtrack_numbers=50)
        gail = A.GAIL(dummy_env, dummy_buffer, config=config)
        gail.train_online(dummy_env, total_iterations=5)

    def test_run_offline_training(self):
        '''
        Check that raising error when calling offline training
        '''
        dummy_env = E.DummyContinuous()
        dummy_buffer = self._create_dummy_buffer(dummy_env)
        gail = A.GAIL(dummy_env, dummy_buffer)

        with pytest.raises(NotImplementedError):
            gail.train_offline([], total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        dummy_buffer = self._create_dummy_buffer(dummy_env)
        gail = A.GAIL(dummy_env, dummy_buffer)

        state = dummy_env.reset()
        state = np.float32(state)
        action = gail.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.GAILConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.GAILConfig(num_steps_per_iteration=-1)
        with pytest.raises(ValueError):
            A.GAILConfig(sigma_kl_divergence_constraint=-0.1)
        with pytest.raises(ValueError):
            A.GAILConfig(maximum_backtrack_numbers=-0.1)
        with pytest.raises(ValueError):
            A.GAILConfig(conjugate_gradient_damping=-0.1)
        with pytest.raises(ValueError):
            A.GAILConfig(conjugate_gradient_iterations=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(vf_epochs=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(vf_batch_size=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(vf_learning_rate=-0.5)
        with pytest.raises(ValueError):
            A.GAILConfig(discriminator_learning_rate=-0.5)
        with pytest.raises(ValueError):
            A.GAILConfig(discriminator_batch_size=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(policy_update_frequency=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(discriminator_update_frequency=-5)
        with pytest.raises(ValueError):
            A.GAILConfig(adversary_entropy_coef=-0.5)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuous()
        dummy_buffer = self._create_dummy_buffer(dummy_env)
        gail = A.GAIL(dummy_env, dummy_buffer)

        gail._v_function_trainer_state = {'v_loss': 0.}
        gail._discriminator_trainer_state = {'reward_loss': 1.}

        latest_iteration_state = gail.latest_iteration_state
        assert 'v_loss' in latest_iteration_state['scalar']
        assert 'reward_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['v_loss'] == 0.
        assert latest_iteration_state['scalar']['reward_loss'] == 1.


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv, generate_dummy_experiences
    pytest.main()
else:
    from .testing_utils import EpisodicEnv, generate_dummy_experiences
