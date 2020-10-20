from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger

import pprint


class IterationStateHook(Hook):
    def __init__(self, writer=None, timing=1000):
        self._timing = timing
        self._writer = writer

    def on_hook_called(self, algorithm):
        logger.info('Iteration state at iteration {}'.format(
            algorithm.iteration_num))

        latest_iteration_state = algorithm.latest_iteration_state

        if 'scalar' in latest_iteration_state:
            logger.info(pprint.pformat(latest_iteration_state['scalar']))

        if self._writer is not None:
            for key, value in latest_iteration_state.items():
                if key == 'scalar':
                    self._writer.write_scalar(algorithm.iteration_num, value)
                if key == 'histogram':
                    self._writer.write_histogram(algorithm.iteration_num, value)
                if key == 'image':
                    self._writer.write_image(algorithm.iteration_num, value)
