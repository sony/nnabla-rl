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


class TestTRPO():
    def setup_method(self):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        atrpo = A.ATRPO(dummy_env)

        assert atrpo.__name__ == 'ATRPO'

    def test_discrete_action_env_unsupported(self):
        '''
        Check that error occurs when training on discrete action env
        '''

        dummy_env = E.DummyDiscrete()
        with pytest.raises(Exception):
            A.ATRPO(dummy_env)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyContinuous()
        dummy_env = EpisodicEnv(dummy_env, min_episode_length=3)

        config = A.ATRPOConfig(num_steps_per_iteration=5,
                               gpu_batch_size=5,
                               pi_batch_size=5,
                               vf_batch_size=2,
                               sigma_kl_divergence_constraint=10.0,
                               maximum_backtrack_numbers=50)
        atrpo = A.ATRPO(dummy_env, config=config)

        atrpo.train_online(dummy_env, total_iterations=5)

    def test_run_offline_training(self):
        '''
        Check that raising error when calling offline training
        '''
        dummy_env = E.DummyContinuous()
        atrpo = A.ATRPO(dummy_env)

        with pytest.raises(NotImplementedError):
            atrpo.train_offline([], total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        atrpo = A.ATRPO(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = atrpo.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.ATRPOConfig(pi_batch_size=-1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(lmb=1.5)
        with pytest.raises(ValueError):
            A.ATRPOConfig(num_steps_per_iteration=-1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(sigma_kl_divergence_constraint=-0.1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(maximum_backtrack_numbers=-0.1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(backtrack_coefficient=-0.1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(conjugate_gradient_damping=-0.1)
        with pytest.raises(ValueError):
            A.ATRPOConfig(conjugate_gradient_iterations=-5)
        with pytest.raises(ValueError):
            A.ATRPOConfig(vf_epochs=-5)
        with pytest.raises(ValueError):
            A.ATRPOConfig(vf_batch_size=-5)
        with pytest.raises(ValueError):
            A.ATRPOConfig(vf_learning_rate=-0.5)
        with pytest.raises(ValueError):
            A.ATRPOConfig(vf_l2_reg_coefficient=-0.5)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuous()
        atrpo = A.ATRPO(dummy_env)

        atrpo._v_function_trainer_state = {'v_loss': 0.}

        latest_iteration_state = atrpo.latest_iteration_state
        assert 'v_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['v_loss'] == 0.


if __name__ == "__main__":
    from testing_utils import EpisodicEnv
    pytest.main()
else:
    from ..testing_utils import EpisodicEnv
