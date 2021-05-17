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


class TestICML2015TRPO(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        trpo = A.ICML2015TRPO(dummy_env)

        assert trpo.__name__ == 'ICML2015TRPO'

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyDiscreteImg()
        dummy_env = EpisodicEnv(dummy_env, min_episode_length=3)
        config = A.ICML2015TRPOConfig(batch_size=5,
                                      gpu_batch_size=2,
                                      num_steps_per_iteration=5,
                                      sigma_kl_divergence_constraint=10.0,
                                      maximum_backtrack_numbers=2)
        trpo = A.ICML2015TRPO(dummy_env, config=config)

        trpo.train_online(dummy_env, total_iterations=1)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''
        dummy_env = E.DummyDiscreteImg()
        trpo = A.ICML2015TRPO(dummy_env)

        with pytest.raises(NotImplementedError):
            trpo.train_offline([], total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        trpo = A.ICML2015TRPO(dummy_env)

        state = dummy_env.reset()
        state = np.empty(dummy_env.observation_space.shape)
        state = np.float32(state)
        action = trpo.compute_eval_action(state)

        assert action.shape == (1,)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.ICML2015TRPOConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.ICML2015TRPOConfig(num_steps_per_iteration=-1)
        with pytest.raises(ValueError):
            A.ICML2015TRPOConfig(sigma_kl_divergence_constraint=-0.1)
        with pytest.raises(ValueError):
            A.ICML2015TRPOConfig(maximum_backtrack_numbers=-0.1)
        with pytest.raises(ValueError):
            A.ICML2015TRPOConfig(conjugate_gradient_damping=-0.1)

    def test_compute_accumulated_reward(self):
        gamma = 0.99
        episode_length = 3
        reward_sequence = np.arange(episode_length)
        gamma_seq = np.array(
            [gamma**i for i in range(episode_length)])
        gamma_seqs = np.zeros((episode_length, episode_length))
        gamma_seqs[0] = gamma_seq
        for i in range(1, episode_length):
            gamma_seqs[i, i:] = gamma_seq[:-i]

        expect = np.sum(reward_sequence*gamma_seqs, axis=1)

        dummy_envinfo = E.DummyContinuous()
        icml2015_trpo = A.ICML2015TRPO(dummy_envinfo)

        accumulated_reward = icml2015_trpo._compute_accumulated_reward(
            reward_sequence, gamma)

        assert expect == pytest.approx(accumulated_reward.flatten())

    def test_compute_accumulated_reward_raise_value_error(self):
        gamma = 0.99
        episode_length = 3
        reward_sequence = np.arange(episode_length).reshape((1, -1))

        dummy_envinfo = E.DummyContinuous()
        icml2015_trpo = A.ICML2015TRPO(dummy_envinfo)

        with pytest.raises(ValueError):
            icml2015_trpo._compute_accumulated_reward(reward_sequence, gamma)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from testing_utils import EpisodicEnv
    pytest.main()
else:
    from .testing_utils import EpisodicEnv
