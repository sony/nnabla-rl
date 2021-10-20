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

import gym
import pytest

from nnabla_rl.environments.dummy import DummyContinuous, DummyContinuousActionGoalEnv
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv

max_episode_steps = 10


class TestGoalConditioned(object):
    def test_not_goal_conditioned_env_unsupported(self):
        env = DummyContinuous(max_episode_steps=max_episode_steps)
        with pytest.raises(ValueError):
            env = GoalConditionedTupleObservationEnv(env)

    def test_observation_space(self):
        env = DummyContinuousActionGoalEnv(max_episode_steps=max_episode_steps)
        env = GoalConditionedTupleObservationEnv(env)

        assert isinstance(env.observation_space, gym.spaces.Tuple)
        assert len(env.observation_space) == 3


if __name__ == "__main__":
    pytest.main()
