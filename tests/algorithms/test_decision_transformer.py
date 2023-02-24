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
from nnabla_rl.replay_buffers import TrajectoryReplayBuffer


class TestDecisionTransformer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg(max_episode_steps=10)
        decision_transformer = A.DecisionTransformer(dummy_env)

        assert decision_transformer.__name__ == 'DecisionTransformer'

    def test_run_online_training(self):
        """Check that error occurs when calling online training."""
        dummy_env = E.DummyDiscreteImg(max_episode_steps=10)

        batch_size = 5
        trajectory_length = 10
        config = A.DecisionTransformerConfig(batch_size=batch_size, max_timesteps=trajectory_length)
        decision_transformer = A.DecisionTransformer(dummy_env, config=config)

        with pytest.raises(NotImplementedError):
            decision_transformer.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        """Check that no error occurs when calling offline training."""
        dummy_env = E.DummyDiscreteImg(max_episode_steps=3)

        trajectory_num = 3
        trajectory_length = 3
        buffer = TrajectoryReplayBuffer()
        for _ in range(trajectory_num):
            trajectory = generate_dummy_trajectory(dummy_env, trajectory_length)
            # Add info required by decision transformer
            trajectory = tuple((s, a, r, done, s_next, {'rtg': 1, 'timesteps': 1})
                               for (s, a, r, done, s_next, *_) in trajectory)
            buffer.append_trajectory(trajectory)

        batch_size = 3
        config = A.DecisionTransformerConfig(batch_size=batch_size, max_timesteps=trajectory_length, context_length=3)
        decision_transformer = A.DecisionTransformer(dummy_env, config=config)

        decision_transformer.train_offline(buffer, total_iterations=2)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg(max_episode_steps=10)

        batch_size = 5
        trajectory_length = 10
        config = A.DecisionTransformerConfig(batch_size=batch_size, max_timesteps=trajectory_length, context_length=10)
        decision_transformer = A.DecisionTransformer(dummy_env, config=config)

        state = dummy_env.reset()
        state = np.float32(state)
        extra_info = {'reward': 0.0}
        action = decision_transformer.compute_eval_action(state, extra_info=extra_info, begin_of_episode=True)
        assert action.shape == (1, )

        action = decision_transformer.compute_eval_action(state, extra_info=extra_info, begin_of_episode=False)
        assert action.shape == (1, )

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(context_length=-1000)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(max_timesteps=-1000)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(grad_clip_norm=-1.0)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(weight_decay=-1.0)
        with pytest.raises(ValueError):
            A.DecisionTransformerConfig(target_return=-1000)

    def test_guess_max_timesteps_infinite_steps(self):
        with pytest.raises(AssertionError):
            dummy_env = E.DummyDiscreteImg()
            A.DecisionTransformer(dummy_env)

    def test_guess_max_timesteps_finite_steps(self):
        dummy_env = E.DummyDiscreteImg(max_episode_steps=10)
        A.DecisionTransformer(dummy_env)

    def test_latest_iteration_state(self):
        """Check that latest iteration state has the keys and values we
        expected."""

        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        trajectory_length = 10
        config = A.DecisionTransformerConfig(batch_size=batch_size, max_timesteps=trajectory_length, context_length=10)
        decision_transformer = A.DecisionTransformer(dummy_env, config=config)

        decision_transformer._decision_transformer_trainer_state = {'loss': 0.}

        latest_iteration_state = decision_transformer.latest_iteration_state
        assert 'loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['loss'] == 0.


if __name__ == "__main__":
    from testing_utils import generate_dummy_trajectory
    pytest.main()
else:
    from ..testing_utils import generate_dummy_trajectory
