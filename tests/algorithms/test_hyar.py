# Copyright 2023 Sony Group Corporation.
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


class TestHyAR(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyHybridEnv()
        hyar = A.HyAR(dummy_env)

        assert hyar.__name__ == 'HyAR'

    def test_discrete_action_env_unsupported(self):
        """Check that error occurs when training on discrete action env."""
        dummy_env = E.DummyDiscrete()
        with pytest.raises(Exception):
            A.HyAR(dummy_env)

    def test_continuous_action_env_unsupported(self):
        """Check that error occurs when training on discrete action env."""
        dummy_env = E.DummyContinuous()
        with pytest.raises(Exception):
            A.HyAR(dummy_env)

    def test_run_online_training(self):
        """Check that no error occurs when calling online training."""
        dummy_env = E.DummyHybridEnv(max_episode_steps=10)
        config = A.HyARConfig(start_timesteps=1,
                              vae_pretrain_episodes=1,
                              vae_pretrain_times=1)
        hyar = A.HyAR(dummy_env, config=config)

        hyar.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """Check that no error occurs when calling offline training."""
        batch_size = 5
        dummy_env = E.DummyHybridEnv(max_episode_steps=10)
        config = A.HyARConfig(start_timesteps=1,
                              vae_pretrain_episodes=1,
                              vae_pretrain_times=1)
        hyar = A.HyAR(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        with pytest.raises(NotImplementedError):
            hyar.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyHybridEnv(max_episode_steps=10)
        hyar = A.HyAR(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        actions = hyar.compute_eval_action(state)

        assert all(np.squeeze(action).shape == space.shape for (action, space) in zip(actions, dummy_env.action_space))

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.HyARConfig(latent_dim=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(embed_dim=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(T=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_pretrain_episodes=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_pretrain_batch_size=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_pretrain_times=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_training_batch_size=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_training_times=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(vae_buffer_size=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(noise_decay_steps=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(initial_exploration_noise=-1.0)
        with pytest.raises(ValueError):
            A.HyARConfig(final_exploration_noise=-0.1)
        with pytest.raises(ValueError):
            A.HyARConfig(latent_select_range=-1)
        with pytest.raises(ValueError):
            A.HyARConfig(latent_select_range=101)
        with pytest.raises(ValueError):
            A.HyARConfig(latent_select_batch_size=-1)

    def test_latest_iteration_state(self):
        """Check that latest iteration state has the keys and values we
        expected."""

        dummy_env = E.DummyHybridEnv()
        hyar = A.HyAR(dummy_env)

        hyar._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}
        hyar._policy_trainer_state = {'pi_loss': 1.}
        hyar._vae_trainer_state = {'encoder_loss': 1., 'kl_loss': 2., 'reconstruction_loss': 3., 'dyn_loss': 4.}

        latest_iteration_state = hyar.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'encoder_loss' in latest_iteration_state['scalar']
        assert 'kl_loss' in latest_iteration_state['scalar']
        assert 'reconstruction_loss' in latest_iteration_state['scalar']
        assert 'dyn_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert latest_iteration_state['scalar']['pi_loss'] == 1.
        assert latest_iteration_state['scalar']['encoder_loss'] == 1.
        assert latest_iteration_state['scalar']['kl_loss'] == 2.
        assert latest_iteration_state['scalar']['reconstruction_loss'] == 3.
        assert latest_iteration_state['scalar']['dyn_loss'] == 4.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
