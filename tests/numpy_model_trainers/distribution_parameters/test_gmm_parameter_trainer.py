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

import numpy as np
import pytest

from nnabla_rl.numpy_model_trainers.distribution_parameters.gmm_parameter_trainer import (GMMParameterTrainer,
                                                                                          GMMParameterTrainerConfig)
from nnabla_rl.numpy_models.distribution_parameters.gmm_parameter import GMMParameter


def _sample_gaussian(means, covs, num_data):
    num_classes, dim = means.shape
    data = []
    for i in range(num_classes):
        class_data = np.random.multivariate_normal(means[i], covs[i], size=num_data // num_classes)
        data.append(class_data)
    return np.concatenate(data, axis=0)


def _is_positive_semidefinite(x):
    return np.all(np.linalg.eigvals(x) >= 0)


class TestGMMParameterTrainer:
    def test_update(self):
        # create dummy dataset
        # sample data
        mean_1 = np.array([3, -4])
        cov_1 = np.array([[0.1, 0.0], [0.0, 0.1]])
        assert _is_positive_semidefinite(cov_1)

        mean_2 = np.array([0.0, 0])
        cov_2 = np.array([[0.75, 0.15], [0.15, 0.75]])
        assert _is_positive_semidefinite(cov_2)

        mean_3 = np.array([-5, 6])
        cov_3 = np.array([[0.5, 0.1], [0.1, 0.5]])
        assert _is_positive_semidefinite(cov_3)

        means = np.stack([mean_1, mean_2, mean_3], axis=0)
        covariances = np.stack([cov_1, cov_2, cov_3], axis=0)

        test_success = False
        # because EM algorithm results strongly depend on the initial parameters, run test multiple times
        for _ in range(5):
            data = _sample_gaussian(means, covariances, 5000)
            trainer = GMMParameterTrainer(
                parameter=GMMParameter.from_data(data, num_classes=3),
                config=GMMParameterTrainerConfig(num_iterations_per_update=1000, threshold=1e-12))
            trainer.update(data)

            order = np.argsort(trainer._parameter._means[:, 0])
            actual_means = trainer._parameter._means[order]
            actual_covariances = trainer._parameter._covariances[order]
            order = np.argsort(means[:, 0])
            expected_means = means[order]
            expected_covariances = covariances[order]

            matched = True
            for actual_mean, actual_covariance, expected_mean, expected_covariance, in zip(
                    actual_means, actual_covariances, expected_means, expected_covariances):
                matched &= np.allclose(actual_mean, expected_mean, atol=1e-1)
                matched &= np.allclose(actual_covariance, expected_covariance, atol=1e-1)
            if matched:
                test_success = True
                break

        assert test_success


if __name__ == "__main__":
    pytest.main()
