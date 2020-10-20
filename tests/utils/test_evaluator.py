import pytest

from unittest import mock

import nnabla_rl.environments as E
import nnabla_rl.algorithms as A
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
