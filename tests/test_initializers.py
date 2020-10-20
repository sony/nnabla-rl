import pytest
from unittest import mock

import numpy as np

import nnabla_rl.initializers as RI


class TestInitializers(object):
    def test_he_normal(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch('nnabla_rl.initializers.calc_normal_std_he_forward', return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeNormal(
                inmaps, outmaps, kernel, factor, mode='fan_in')
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

        with mock.patch('nnabla_rl.initializers.calc_normal_std_he_backward', return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeNormal(
                inmaps, outmaps, kernel, factor, mode='fan_out')
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_lecun_normal(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch('nnabla_rl.initializers.calc_normal_std_he_forward', return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.LeCunNormal(
                inmaps, outmaps, kernel, factor, mode='fan_in')
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_he_uniform(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch('nnabla_rl.initializers.calc_uniform_lim_he_forward', return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeUniform(
                inmaps, outmaps, kernel, factor, mode='fan_in')
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

        with mock.patch('nnabla_rl.initializers.calc_uniform_lim_he_backward', return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeUniform(
                inmaps, outmaps, kernel, factor, mode='fan_out')
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_he_normal_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.HeNormal(inmaps, outmaps, kernel, factor, mode='fan_unknown')

    def test_he_uniform_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.HeUniform(inmaps, outmaps, kernel, factor, mode='fan_unknown')

    def test_lecun_normal_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.LeCunNormal(inmaps, outmaps, kernel, factor, mode='fan_out')

        with pytest.raises(NotImplementedError):
            RI.LeCunNormal(inmaps, outmaps, kernel, factor, mode='fan_unknown')

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3*i, 5*i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_normal_std_he_forward(self, inmap, outmap, kernel, factor):
        n = inmap * kernel[0] * kernel[1]
        expected = np.sqrt(factor / n)
        actual = RI.calc_normal_std_he_forward(
            inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3*i, 5*i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_normal_std_he_backward(self, inmap, outmap, kernel, factor):
        n = outmap * kernel[0] * kernel[1]
        expected = np.sqrt(factor / n)
        actual = RI.calc_normal_std_he_backward(
            inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3*i, 5*i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_uniform_lim_he_forward(self, inmap, outmap, kernel, factor):
        n = inmap * kernel[0] * kernel[1]
        expected = np.sqrt((3.0 * factor) / n)
        actual = RI.calc_uniform_lim_he_forward(
            inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3*i, 5*i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_uniform_lim_he_backward(self, inmap, outmap, kernel, factor):
        n = outmap * kernel[0] * kernel[1]
        expected = np.sqrt((3.0 * factor) / n)
        actual = RI.calc_uniform_lim_he_backward(
            inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("std", [0.5 * i for i in range(1, 10)])
    @pytest.mark.parametrize("axis", [i for i in range(0, 2)])
    def test_normc_initializer(self, std, axis):
        rng = np.random
        rng.seed(0)
        initializer = RI.NormcInitializer(std=std, axis=axis, rng=rng)

        shape = (5, 5)
        actual = initializer(shape)

        np.random.seed(0)
        expected = np.random.randn(*shape)
        expected *= std / np.sqrt(np.square(expected).sum(axis=axis, keepdims=True))

        np.testing.assert_almost_equal(actual, expected)


if __name__ == "__main__":
    pytest.main()
