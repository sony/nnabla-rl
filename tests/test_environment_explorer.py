# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from nnabla_rl.environment_explorer import _is_end_of_episode, _sample_action
from nnabla_rl.environments.dummy import (
    DummyContinuous,
    DummyDiscrete,
    DummyTupleActionContinuous,
    DummyTupleActionDiscrete,
    DummyTupleMixed,
)
from nnabla_rl.environments.environment_info import EnvironmentInfo


class TestEnvironmentExplorer(object):
    @pytest.mark.parametrize("done", [True, False])
    @pytest.mark.parametrize("timelimit", [True, False])
    @pytest.mark.parametrize("timelimit_as_terminal", [True, False])
    def test_is_end_of_episode(self, done, timelimit, timelimit_as_terminal):
        end_of_episode = _is_end_of_episode(done, timelimit, timelimit_as_terminal)
        if not done:
            assert end_of_episode is False
        else:
            # All the case that done == True
            if timelimit and timelimit_as_terminal:
                assert end_of_episode is True
            elif timelimit and not timelimit_as_terminal:
                assert end_of_episode is False
            elif not timelimit:
                assert end_of_episode is True
            else:
                raise RuntimeError

    @pytest.mark.parametrize(
        "env",
        [
            DummyContinuous(),
            DummyDiscrete(),
            DummyTupleActionContinuous(),
            DummyTupleActionDiscrete(),
            DummyTupleMixed(),
        ],
    )
    def test_sample_action(self, env):
        env_info = EnvironmentInfo.from_env(env)
        action, *_ = _sample_action(env, env_info)

        if env_info.is_tuple_action_env():
            for a, space in zip(action, env_info.action_space):
                if isinstance(space, gym.spaces.Discrete):
                    assert a.shape == (1,)
                else:
                    assert a.shape == space.shape
        else:
            if isinstance(env_info.action_space, gym.spaces.Discrete):
                assert action.shape == (1,)
            else:
                assert action.shape == env_info.action_space.shape or action.shape == (1,)


if __name__ == "__main__":
    pytest.main()
