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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import nnabla_rl as rl
import nnabla_rl.environments as E
from nnabla_rl.algorithm import Algorithm, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer


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

            @eval_api
            def compute_eval_action(self, x):
                assert rl.is_eval_scope()

            def _setup_solver(self):
                pass

            def _models(self):
                pass

            def _solvers(self):
                pass

        eval_scope_check = EvalScopeCheck()
        eval_scope_check.compute_eval_action(x=np.empty(shape=(1, 5)))


if __name__ == "__main__":
    pytest.main()
