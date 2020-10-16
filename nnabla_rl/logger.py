from contextlib import contextmanager

import logging

logger = logging.getLogger('nnabla_rl')
logger.disabled = False


@contextmanager
def enable_logging(level=logging.INFO):
    return _switch_logability(disabled=False, level=level)


@contextmanager
def disable_logging(level=logging.INFO):
    return _switch_logability(disabled=True, level=level)


def _switch_logability(disabled, level=logging.INFO):
    global logger
    previous_level = logger.level
    previous_status = logger.disabled
    try:
        logger.disabled = disabled
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(previous_level)
        logger.disabled = previous_status
