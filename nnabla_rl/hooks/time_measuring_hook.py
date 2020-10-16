import time

from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class TimeMeasuringHook(Hook):
    def __init__(self, timing=1):
        super(TimeMeasuringHook, self).__init__(timing=timing)
        self._prev_time = time.time()

    def on_hook_called(self, algorithm):
        current_time = time.time()

        logger.info("time spent since previous hook: {} seconds".format(
            current_time - self._prev_time))

        self._prev_time = current_time
