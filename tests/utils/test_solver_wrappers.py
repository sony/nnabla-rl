# Copyright 2022 Sony Group Corporation.
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

from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm, AutoWeightDecay, SolverWrapper


class TestSolverWrappers():
    def test_auto_clip_grad_by_norm(self):
        norm = 10.0
        solver_mock = mock.MagicMock()
        update_mock = mock.MagicMock()
        clip_grad_by_norm_mock = mock.MagicMock()
        solver = SolverWrapper(solver_mock)
        solver.update = update_mock
        solver.clip_grad_by_norm = clip_grad_by_norm_mock
        solver = AutoClipGradByNorm(solver, norm)

        solver.update()

        update_mock.assert_called_once()
        clip_grad_by_norm_mock.assert_called_once()

    def test_auto_weight_decay(self):
        norm = 10.0
        solver_mock = mock.MagicMock()
        update_mock = mock.MagicMock()
        weight_decay_mock = mock.MagicMock()
        solver = SolverWrapper(solver_mock)
        solver.update = update_mock
        solver.weight_decay = weight_decay_mock
        solver = AutoWeightDecay(solver, norm)

        solver.update()

        update_mock.assert_called_once()
        weight_decay_mock.assert_called_once()


if __name__ == '__main__':
    pytest.main()
