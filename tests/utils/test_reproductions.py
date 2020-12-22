import pytest
from unittest.mock import Mock

import nnabla.initializer as NI
import numpy as np

import nnabla_rl.utils.reproductions as reproductions


class TestReproductions():
    def test_set_global_seed(self):
        seed = 0

        env_mock = Mock()
        reproductions.set_global_seed(seed, env=env_mock)
        random_interger1 = np.random.randint(low=10)
        initializer1 = NI.UniformInitializer()
        init_param1 = initializer1(shape=(10, 10))

        env_mock.seed.assert_called_once_with(seed)

        env_mock.reset_mock()
        reproductions.set_global_seed(seed, env=env_mock)
        random_interger2 = np.random.randint(low=10)
        initializer2 = NI.UniformInitializer()
        init_param2 = initializer2(shape=(10, 10))

        env_mock.seed.assert_called_once_with(seed)
        assert random_interger1 == random_interger2
        assert np.allclose(init_param1, init_param2)


if __name__ == '__main__':
    pytest.main()
