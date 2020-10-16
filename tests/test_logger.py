import pytest

import nnabla_rl.logger as logger


class TestLogger(object):
    def test_enable_logging(self):
        # disable logging
        logger.logger.disabled = True
        assert logger.logger.disabled
        with logger.enable_logging():
            assert not logger.logger.disabled
        assert logger.logger.disabled

    def test_disable_logging(self):
        logger.logger.disabled = False
        assert not logger.logger.disabled
        with logger.disable_logging():
            assert logger.logger.disabled
        assert not logger.logger.disabled


if __name__ == "__main__":
    pytest.main()
