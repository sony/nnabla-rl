import pytest

import numpy as np
import scipy.stats as stats

from unittest import mock

import nnabla as nn
import nnabla.functions as F

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
