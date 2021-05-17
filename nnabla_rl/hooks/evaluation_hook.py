# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import numpy as np

from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger
from nnabla_rl.utils.evaluator import EpisodicEvaluator


class EvaluationHook(Hook):
    '''
    Hook to run evaluation during training.

    Args:
        env (gym.Env): Environment to run the evaluation
        evaluator (Callable[[nnabla_rl.algorithm.Algorithm, gym.Env], List[float]]):
            Evaluator which runs the actual evaluation.
            Defaults to :py:class:`EpisodicEvaluator <nnabla_rl.utils.evaluator.EpisodicEvaluator>`.
        timing (int): Evaluation interval. Defaults to 1000 iteration.
        writer (nnabla_rl.writer.Writer, optional): Writer instance to save/print the evaluation results.
            Defaults to None.
    '''

    def __init__(self, env, evaluator=EpisodicEvaluator(), timing=1000, writer=None):
        super(EvaluationHook, self).__init__(timing=timing)
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
