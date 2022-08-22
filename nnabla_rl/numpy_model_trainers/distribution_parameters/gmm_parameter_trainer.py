# Copyright 2022 Sony Group Corporation.
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

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from nnabla_rl.distributions.gmm import NumpyGMM, compute_mixing_coefficient, compute_responsibility, logsumexp
from nnabla_rl.logger import logger
from nnabla_rl.numpy_model_trainers.numpy_model_trainer import NumpyModelTrainer, NumpyModelTrainerConfig
from nnabla_rl.numpy_models.distribution_parameters.gmm_parameter import GMMParameter


@dataclass
class GMMParameterTrainerConfig(NumpyModelTrainerConfig):
    '''GMM Trainer Configuration'''

    num_iterations_per_update: int = 1000
    threshold: float = 1e-12

    def __post_init__(self):
        super(GMMParameterTrainerConfig, self).__post_init__()


class GMMParameterTrainer(NumpyModelTrainer):
    '''Gaussian Mixture Model Trainer with EM algorithm'''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _parameter: GMMParameter
    _config: GMMParameterTrainerConfig

    def __init__(self,
                 parameter: GMMParameter,
                 config: GMMParameterTrainerConfig = GMMParameterTrainerConfig()):
        super(GMMParameterTrainer, self).__init__(config)
        self._parameter = parameter

    def update(self, data: np.ndarray) -> None:
        '''update the parameters of gmm

        Args:
            data (np.ndarray): data, shape(num_data, dim)
        '''
        prev_log_likelihood = -float('inf')
        for _ in range(self._config.num_iterations_per_update):
            probs, responsibility = self._e_step(data)

            # for checking convergence
            log_likelihood = np.sum(logsumexp(np.log(probs), axis=1, keepdims=True))
            if self._has_converged(log_likelihood, prev_log_likelihood):
                break

            prev_log_likelihood = log_likelihood
            self._m_step(data, responsibility)

    def _e_step(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return compute_responsibility(data, NumpyGMM.from_gmm_parameter(self._parameter))

    def _m_step(self, data: np.ndarray, responsibility: np.ndarray) -> None:
        _, dim = data.shape
        new_mixing_coefficients = compute_mixing_coefficient(responsibility)

        # adding small value to avoid computational error
        # nk.shape = (num_classes, )
        nk = np.exp(logsumexp(np.log(responsibility), axis=0)) + 1e-8
        # new_means.shape = (num_classes, dim)
        new_means = np.dot(responsibility.T, data) / nk[:, np.newaxis]
        new_covariances = np.zeros((self._parameter._num_classes, dim, dim))

        for i in range(self._parameter._num_classes):
            diff = data - new_means[i, :]
            variance = np.dot(responsibility[:, i] * diff.T, diff) / nk[i]
            new_covariances[i] = variance + 1e-8 * np.eye(dim)  # for regularization

        self._parameter.update_parameter(new_means, new_covariances, new_mixing_coefficients)

    def _has_converged(self, new_log_likelihood, prev_log_likelihood):
        if np.abs(new_log_likelihood - prev_log_likelihood) < self._config.threshold:
            logger.debug('GMM converged before reaching max iterations')
            return True
        else:
            return False
