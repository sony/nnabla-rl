import pytest

from unittest.mock import patch
from unittest.mock import MagicMock

from nnabla_rl.algorithm import Algorithm
from nnabla_rl.replay_buffer import ReplayBuffer
import nnabla_rl.environments as E
import nnabla_rl as rl


class TestAlgorithm(object):
    @patch.multiple(Algorithm, __abstractmethods__=set())
    def test_resume_online_training(self):
        env = E.DummyContinuous()
        algorithm = Algorithm(env)
        algorithm._run_online_training_iteration = MagicMock()

        total_iterations = 10
        algorithm.train(env, total_iterations=total_iterations)
        assert algorithm._run_online_training_iteration.call_count == total_iterations

        algorithm.train(env, total_iterations=total_iterations)
        assert algorithm._run_online_training_iteration.call_count == total_iterations * 2

    @patch.multiple(Algorithm, __abstractmethods__=set())
    def test_resume_offline_training(self):
        env = E.DummyContinuous()
        algorithm = Algorithm(env)
        algorithm._run_offline_training_iteration = MagicMock()

        total_iterations = 10
        # It is ok to pass empty buffer because this algorithm never use the buffer
        buffer = ReplayBuffer()
        algorithm.train(buffer, total_iterations=total_iterations)
        assert algorithm._run_offline_training_iteration.call_count == total_iterations

        algorithm.train(buffer, total_iterations=total_iterations)
        assert algorithm._run_offline_training_iteration.call_count == total_iterations * 2

    def test_eval_scope(self):
        class EvalScopeCheck(Algorithm):
            def __init__(self):
                pass

            def _run_online_training_iteration(self, env):
                pass

            def _run_offline_training_iteration(self, buffer):
                pass

            def compute_eval_action(self, x):
                pass

            def _build_training_graph(self):
                assert not rl.is_eval_scope()

            def _build_evaluation_graph(self):
                assert rl.is_eval_scope()

            def _setup_solver(self):
                pass

            def _models(self):
                pass

            def _solvers(self):
                pass

        eval_scope_check = EvalScopeCheck()
        eval_scope_check._build_evaluation_graph()
        eval_scope_check._build_training_graph()


if __name__ == "__main__":
    pytest.main()
