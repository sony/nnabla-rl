from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger
from nnabla_rl.utils.evaluator import EpisodicEvaluator

import numpy as np


class EvaluationHook(Hook):
    def __init__(self, env, evaluator=EpisodicEvaluator(), timing=1000, writer=None):
        super(EvaluationHook, self).__init__(timing=timing)
        '''
        Hook to run evaluation during training.
        '''
        self._env = env
        self._evaluator = evaluator
        self._timing = timing
        self._writer = writer

    def on_hook_called(self, algorithm):
        iteration_num = algorithm.iteration_num
        logger.info(
            'Starting evaluation at iteration {}.'.format(iteration_num))
        returns = self._evaluator(algorithm, self._env)
        mean = np.mean(returns)
        std_dev = np.std(returns)
        median = np.median(returns)
        logger.info('Evaluation results at iteration {}. mean: {} +/- std: {}, median: {}'.format(
            iteration_num, mean, std_dev, median))

        if self._writer is not None:
            minimum = np.min(returns)
            maximum = np.max(returns)
            # From python 3.6 or above, the dictionary preserves insertion order
            scalar_results = {}
            scalar_results['mean'] = mean
            scalar_results['std_dev'] = std_dev
            scalar_results['min'] = minimum
            scalar_results['max'] = maximum
            scalar_results['median'] = median
            self._writer.write_scalar(algorithm.iteration_num, scalar_results)

            histogram_results = {}
            histogram_results['returns'] = returns
            self._writer.write_histogram(algorithm.iteration_num, histogram_results)
