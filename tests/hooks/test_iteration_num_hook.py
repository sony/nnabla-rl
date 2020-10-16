from nnabla_rl.hooks import IterationNumHook
from nnabla_rl.writer import Writer

from unittest import mock
import pytest
from nnabla_rl.logger import logger


class TestIterationStateHook():
    def test_call(self):
        dummy_algorithm = mock.MagicMock()

        hook = IterationNumHook(timing=1)

        with mock.patch.object(logger, 'info') as mock_logger:
            for i in range(10):
                dummy_algorithm.iteration_num = i
                hook(dummy_algorithm)
            assert mock_logger.call_count == 10


if __name__ == "__main__":
    pytest.main()
