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
