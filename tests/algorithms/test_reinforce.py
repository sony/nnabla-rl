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


class TestREINFORCE(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscrete()
        reinforce = A.REINFORCE(dummy_env)

        assert reinforce.__name__ == 'REINFORCE'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscrete()
        dummy_env = EpisodicEnv(dummy_env)
        reinforce = A.REINFORCE(dummy_env)
        reinforce.train_online(dummy_env, total_iterations=1)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        dummy_env = E.DummyDiscrete()
        reinforce = A.REINFORCE(dummy_env)

        with pytest.raises(NotImplementedError):
            reinforce.train_offline([], total_iterations=2)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.REINFORCEConfig(reward_scale=-0.1)
        with pytest.raises(ValueError):
            A.REINFORCEConfig(num_rollouts_per_train_iteration=-1)
        with pytest.raises(ValueError):
            A.REINFORCEConfig(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.REINFORCEConfig(clip_grad_norm=-0.1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscrete()
        reinforce = A.REINFORCE(dummy_env)

        reinforce._policy_trainer_state = {'pi_loss': 0.}

        latest_iteration_state = reinforce.latest_iteration_state
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['pi_loss'] == 0.


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv
    pytest.main()
else:
    from .testing_utils import EpisodicEnv
