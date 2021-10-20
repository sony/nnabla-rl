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
from nnabla_rl.environments.dummy import DummyContinuous, DummyContinuousActionGoalEnv, DummyDiscreteActionGoalEnv
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.replay_buffer import ReplayBuffer

max_episode_steps = 10
num_episode = 4
num_experiences = max_episode_steps * num_episode


class TestHER(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        her = A.HER(dummy_env)

        assert her.__name__ == 'HER'

    def test_not_goal_conditioned_env_unsupported(self):
        '''
        Check that error occurs when training on not goal-conditioned env
        '''

        dummy_env = DummyContinuous(max_episode_steps=max_episode_steps)
        config = A.HERConfig()
        with pytest.raises(Exception):
            A.HER(dummy_env, config=config)

    def test_discrete_action_goal_conditioned_env_unsupported(self):
        '''
        Check that error occurs when training on discrete action goal-conditioned env
        '''

        dummy_env = DummyDiscreteActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        config = A.HERConfig()
        with pytest.raises(Exception):
            A.HER(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        batch_size = 5
        config = A.HERConfig(batch_size=batch_size, start_timesteps=5)
        her = A.HER(dummy_env, config=config)

        her.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        batch_size = 5
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        config = A.HERConfig(batch_size=batch_size)
        her = A.HER(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        her.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        her = A.HER(dummy_env)

        state = dummy_env.reset()
        action = her.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        her = A.HER(dummy_env)

        her._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}
        her._policy_trainer_state = {'pi_loss': 1.}

        latest_iteration_state = her.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert latest_iteration_state['scalar']['pi_loss'] == 1.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
