from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class IterationNumHook(Hook):
    def __init__(self, timing=1):
        super(IterationNumHook, self).__init__(timing=timing)

    def on_hook_called(self, algorithm):
        logger.info("current iteration: {}".format(algorithm.iteration_num))
