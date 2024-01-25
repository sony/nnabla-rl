# Copyright 2024 Sony Group Corporation.
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

from nnabla_rl.environments.amp_env import AMPEnv, AMPGoalEnv, TaskResult


class DummyAMPEnv(AMPEnv):
    def __init__(self, dummy_transition, dummy_task_result, dummy_is_valid_episode, dummy_expert_experience) -> None:
        super().__init__()
        self._dummy_transition = dummy_transition
        self._dummy_task_result = dummy_task_result
        self._dummy_is_valid_episode = dummy_is_valid_episode
        self._dummy_expert_experience = dummy_expert_experience

    def _step(self, action):
        return self._dummy_transition

    def task_result(self, state, reward, done, info):
        return self._dummy_task_result

    def is_valid_episode(self, state, reward, done, info):
        return self._dummy_is_valid_episode

    def expert_experience(self, state, reward, done, info):
        return self._dummy_expert_experience


class TestAMPEnv():
    def test_step(self):
        state = np.array([1.1])
        reward = 5.0
        done = True
        info = {}
        dummy_transition = (state, reward, done, info)
        dummy_task_result = TaskResult(1)
        dummy_is_valid_episode = False

        expert_state = np.array([1.2])
        expert_action = np.array([1.5])
        expert_next_state = np.array([1.7])
        expert_reward = 0.5
        expert_non_terminal = 1.0
        expert_info = {}
        dummy_expert_experience = (expert_state, expert_action, expert_reward,
                                   expert_non_terminal, expert_next_state, expert_info)

        env = DummyAMPEnv(dummy_transition, dummy_task_result, dummy_is_valid_episode, dummy_expert_experience)
        env.reset()
        actual_state, actual_reward, actual_done, actual_info = env.step(np.random.rand(5))

        # Check not overwrite original step
        assert np.allclose(actual_state, state)
        assert actual_reward == actual_reward
        assert actual_done == done

        # Check correct info is included
        assert "task_result" in actual_info and actual_info["task_result"] == dummy_task_result

        assert "valid_episode" in actual_info and actual_info["valid_episode"] == dummy_is_valid_episode

        assert "expert_experience" in actual_info
        assert np.allclose(actual_info["expert_experience"][0], expert_state)
        assert np.allclose(actual_info["expert_experience"][1], expert_action)
        assert actual_info["expert_experience"][2] == expert_reward
        assert np.allclose(actual_info["expert_experience"][3], expert_non_terminal)
        assert np.allclose(actual_info["expert_experience"][4], expert_next_state)


class DummyAMPGoalEnv(AMPGoalEnv):
    def __init__(self, dummy_transition, dummy_task_result, dummy_is_valid_episode, dummy_expert_experience) -> None:
        super().__init__()
        self._dummy_transition = dummy_transition
        self._dummy_task_result = dummy_task_result
        self._dummy_is_valid_episode = dummy_is_valid_episode
        self._dummy_expert_experience = dummy_expert_experience

    def _step(self, action):
        return self._dummy_transition

    def task_result(self, state, reward, done, info):
        return self._dummy_task_result

    def is_valid_episode(self, state, reward, done, info):
        return self._dummy_is_valid_episode

    def expert_experience(self, state, reward, done, info):
        return self._dummy_expert_experience


class TestAMPGoalEnv():
    def test_step(self):
        state = np.array([1.1])
        reward = 5.0
        done = True
        info = {}
        dummy_transition = (state, reward, done, info)
        dummy_task_result = TaskResult(1)
        dummy_is_valid_episode = False

        expert_state = np.array([1.2])
        expert_action = np.array([1.5])
        expert_next_state = np.array([1.7])
        expert_reward = 0.5
        expert_non_terminal = 1.0
        expert_info = {}
        dummy_expert_experience = (expert_state, expert_action, expert_reward,
                                   expert_non_terminal, expert_next_state, expert_info)

        env = DummyAMPEnv(dummy_transition, dummy_task_result, dummy_is_valid_episode, dummy_expert_experience)
        env.reset()
        actual_state, actual_reward, actual_done, actual_info = env.step(np.random.rand(5))

        # Check not overwrite original step
        assert np.allclose(actual_state, state)
        assert actual_reward == actual_reward
        assert actual_done == done

        # Check correct info is included
        assert "task_result" in actual_info and actual_info["task_result"] == dummy_task_result

        assert "valid_episode" in actual_info and actual_info["valid_episode"] == dummy_is_valid_episode

        assert "expert_experience" in actual_info
        assert np.allclose(actual_info["expert_experience"][0], expert_state)
        assert np.allclose(actual_info["expert_experience"][1], expert_action)
        assert actual_info["expert_experience"][2] == expert_reward
        assert np.allclose(actual_info["expert_experience"][3], expert_non_terminal)
        assert np.allclose(actual_info["expert_experience"][4], expert_next_state)


if __name__ == "__main__":
    pytest.main()
