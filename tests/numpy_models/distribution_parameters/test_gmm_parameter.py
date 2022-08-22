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

from nnabla_rl.numpy_models.distribution_parameters.gmm_parameter import GMMParameter


class TestGMMParameter:
    def test_update_parameters(self):
        means = np.array([[3.0], [5.0]])
        covariances = np.array([[[0.4]], [[0.1]]])
        mixing_coefficients = np.array([0.4, 0.6])
        gmm_param = GMMParameter(means, covariances, mixing_coefficients)

        new_means = np.array([[20.0], [10.0]])
        new_covariances = np.array([[[1.8]], [[1.5]]])
        new_mixing_coefficients = np.array([0.9, 0.1])
        gmm_param.update_parameter(new_means, new_covariances, new_mixing_coefficients)

        assert np.allclose(new_means, gmm_param._means)
        assert np.allclose(new_covariances, gmm_param._covariances)
        assert np.allclose(new_mixing_coefficients, gmm_param._mixing_coefficients)


if __name__ == "__main__":
    pytest.main()
