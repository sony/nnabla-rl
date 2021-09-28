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

from nnabla_rl.environments.dummy import DummyContinuousActionGoalEnv
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.replay_buffers.hindsight_replay_buffer import HindsightReplayBuffer

max_episode_steps = 10
num_episode = 4
num_experiences = max_episode_steps * num_episode


class TestHindsightReplayBuffer(object):
    def setup_method(self, method):
        np.random.seed(0)

    def test_unsupported_env(self):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        buffer = HindsightReplayBuffer(reward_function=dummy_env.compute_reward,
                                       capacity=100)
        experiences = self._generate_experiences(dummy_env)

        with pytest.raises(RuntimeError):
            buffer.append_all(experiences)

    def test_extract_end_index_of_episode(self):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        buffer = HindsightReplayBuffer(reward_function=dummy_env.compute_reward,
                                       capacity=100)
        for _ in range(num_episode):
            experiences = self._generate_experiences(dummy_env)
            buffer.append_all(experiences)

        for i in range(num_episode):
            episode_start_index = i * max_episode_steps
            end_index_of_episode = self._extract_end_index_of_episode(buffer, episode_start_index)
            gt_end_index_of_episode = (i + 1) * max_episode_steps - 1
            assert end_index_of_episode == gt_end_index_of_episode

    @pytest.mark.parametrize('index', [np.random.randint(num_experiences) for _ in range(10)])
    def test_select_future_index(self, index):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        buffer = HindsightReplayBuffer(reward_function=dummy_env.compute_reward,
                                       capacity=100)
        for _ in range(num_episode):
            experiences = self._generate_experiences(dummy_env)
            buffer.append_all(experiences)

        end_index_of_episode = self._extract_end_index_of_episode(buffer, index)
        future_index = buffer._select_future_index(index, end_index_of_episode)
        assert ((index <= future_index) and (future_index <= end_index_of_episode))

    @pytest.mark.parametrize('index', [np.random.randint(num_experiences) for _ in range(10)])
    def test_replace_goal(self, index):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        buffer = HindsightReplayBuffer(reward_function=dummy_env.compute_reward,
                                       capacity=100)
        for _ in range(num_episode):
            experiences = self._generate_experiences(dummy_env)
            buffer.append_all(experiences)

        experience = buffer.__getitem__(index)
        end_index_of_episode = self._extract_end_index_of_episode(buffer, index)
        future_index = buffer._select_future_index(index, end_index_of_episode)
        future_experience = buffer.__getitem__(future_index)
        new_experience = buffer._replace_goal(experience, future_experience)

        assert(np.allclose(new_experience[0][1], future_experience[4][2]))
        assert(np.allclose(new_experience[4][1], future_experience[4][2]))

    @pytest.mark.parametrize('index', [np.random.randint(num_experiences) for _ in range(10)])
    def test_replace_goal_with_same_index(self, index):
        dummy_env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        buffer = HindsightReplayBuffer(reward_function=dummy_env.compute_reward,
                                       capacity=100)
        for _ in range(num_episode):
            experiences = self._generate_experiences(dummy_env)
            buffer.append_all(experiences)

        experience = buffer.__getitem__(index)
        future_experience = buffer.__getitem__(index)
        new_experience = buffer._replace_goal(experience, future_experience)

        assert(np.allclose(new_experience[0][1], future_experience[4][2]))
        assert(np.allclose(new_experience[4][1], future_experience[4][2]))
        assert(new_experience[2] == 1.0)

    def _generate_experiences(self, env):
        experiences = []
        s = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            n_s, r, done, info = env.step(a)
            non_terminal = 0 if done else 1
            experience = (s, a, r, non_terminal, n_s, info)
            experiences.append(experience)
            s = n_s
        return experiences

    def _extract_end_index_of_episode(self, buffer, item_index):
        _, _, _, _, _, info = buffer[item_index]
        index_in_episode = info['index_in_episode']
        episode_end_index = int(info['episode_end_index'])
        distance_to_end = episode_end_index - index_in_episode

        return distance_to_end + item_index


if __name__ == "__main__":
    pytest.main()
