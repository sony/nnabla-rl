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

from unittest import mock

import pytest

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator


class TestEpisodicEvaluator(object):
    def test_evaluation(self):
        run_per_evaluation = 10
        dummy_env = E.DummyContinuous()
        dummy_algorithm = A.Dummy(dummy_env)
        evaluator = EpisodicEvaluator(run_per_evaluation=run_per_evaluation)

        dummy_env.reset = mock.MagicMock()
        dummy_env.reset.return_value = None

        dummy_env.step = mock.MagicMock()
        dummy_env.step.return_value = (None, None, True, None)

        dummy_algorithm.compute_eval_action = mock.MagicMock()
        dummy_algorithm.return_value = None

        evaluator(dummy_algorithm, dummy_env)

        assert dummy_env.reset.call_count == run_per_evaluation
        assert dummy_env.step.call_count == run_per_evaluation
        assert dummy_algorithm.compute_eval_action.call_count == run_per_evaluation


class TestTimestepEvaluator(object):
    def test_evaluation(self):
        num_timesteps = 100
        max_episode_length = 10
        dummy_env = E.DummyAtariEnv(
            done_at_random=False, max_episode_length=max_episode_length)
        evaluator = TimestepEvaluator(num_timesteps=num_timesteps)

        dummy_algorithm = A.Dummy(dummy_env)
        dummy_algorithm.compute_eval_action = mock.MagicMock()

        returns = evaluator(dummy_algorithm, dummy_env)

        assert len(returns) == num_timesteps // max_episode_length
        assert dummy_algorithm.compute_eval_action.call_count == num_timesteps + 1

    def test_timestep_limit(self):
        num_timesteps = 113
        max_episode_length = 10
        dummy_env = E.DummyAtariEnv(
            done_at_random=False, max_episode_length=max_episode_length)
        evaluator = TimestepEvaluator(num_timesteps=num_timesteps)

        dummy_algorithm = A.Dummy(dummy_env)
        dummy_algorithm.compute_eval_action = mock.MagicMock()

        returns = evaluator(dummy_algorithm, dummy_env)

        assert len(returns) == num_timesteps // max_episode_length
        assert dummy_algorithm.compute_eval_action.call_count == num_timesteps + 1


if __name__ == '__main__':
    pytest.main()
