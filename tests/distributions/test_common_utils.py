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
import pytest
import scipy.stats as stats

import nnabla as nn
import nnabla_rl.distributions.common_utils as common_utils


class TestCommonUtils(object):
    def setup_method(self, method):
        nn.clear_parameters()
        np.random.seed(0)

    def test_gaussian_log_prob(self):
        x = np.random.randn(1, 4) * 10
        mean = np.random.randn(1, 4)
        ln_var = np.random.randn(1, 4) * 5.0
        var = np.exp(ln_var)

        actual = common_utils.gaussian_log_prob(nn.Variable.from_numpy_array(x),
                                                nn.Variable.from_numpy_array(mean),
                                                nn.Variable.from_numpy_array(var),
                                                nn.Variable.from_numpy_array(ln_var))
        actual.forward()
        actual = actual.d

        # FIXME: avoid using scipy
        gaussian = stats.multivariate_normal(mean=mean.squeeze(), cov=np.diag(var.squeeze()))
        expected = gaussian.logpdf(x)

        assert np.allclose(expected, actual)


if __name__ == "__main__":
    pytest.main()
