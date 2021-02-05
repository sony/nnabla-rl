# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import pytest

import pathlib

import numpy as np

import nnabla_rl.algorithms as A
from nnabla_rl.utils.serializers import load_snapshot


class TestLoadSnapshot(object):
    def test_load_snapshot(self):
        snapshot_path = pathlib.Path('test_resources/utils/ddpg-snapshot')
        ddpg = load_snapshot(snapshot_path)

        assert isinstance(ddpg, A.DDPG)
        assert ddpg.iteration_num == 10000
        assert np.isclose(ddpg._params.tau, 0.05)
        assert np.isclose(ddpg._params.gamma, 0.99)
        assert np.isclose(ddpg._params.learning_rate, 0.001)
        assert np.isclose(ddpg._params.batch_size, 100)
        assert np.isclose(ddpg._params.start_timesteps, 200)
        assert ddpg._params.replay_buffer_size == 1000000


if __name__ == '__main__':
    pytest.main()
