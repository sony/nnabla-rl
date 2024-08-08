# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from unittest import mock

import numpy as np
import pytest

import nnabla_rl.initializers as RI


class TestInitializers(object):
    def test_he_normal(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch("nnabla_rl.initializers.calc_normal_std_he_forward", return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeNormal(inmaps, outmaps, kernel, factor, mode="fan_in")
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

        with mock.patch("nnabla_rl.initializers.calc_normal_std_he_backward", return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeNormal(inmaps, outmaps, kernel, factor, mode="fan_out")
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_lecun_normal(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch("nnabla_rl.initializers.calc_normal_std_he_forward", return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.LeCunNormal(inmaps, outmaps, kernel, factor, mode="fan_in")
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_he_uniform(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with mock.patch("nnabla_rl.initializers.calc_uniform_lim_he_forward", return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeUniform(inmaps, outmaps, kernel, factor, mode="fan_in")
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

        with mock.patch("nnabla_rl.initializers.calc_uniform_lim_he_backward", return_value=True) as mock_calc:
            mock_calc.return_value = 10

            initializer = RI.HeUniform(inmaps, outmaps, kernel, factor, mode="fan_out")
            initializer(shape=(5, 5))
            mock_calc.assert_called_once_with(inmaps, outmaps, kernel, factor)

    def test_he_normal_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.HeNormal(inmaps, outmaps, kernel, factor, mode="fan_unknown")

    def test_he_uniform_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.HeUniform(inmaps, outmaps, kernel, factor, mode="fan_unknown")

    def test_he_uniform_with_rng(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0
        rng = np.random.RandomState(0)

        initializer1 = RI.HeUniform(inmaps, outmaps, kernel, factor, rng=rng)
        params1 = initializer1(shape=(5, 5))

        rng = np.random.RandomState(0)
        initializer2 = RI.HeUniform(inmaps, outmaps, kernel, factor, rng=rng)
        params2 = initializer2(shape=(5, 5))

        assert np.allclose(params1, params2)

        rng = np.random.RandomState(1)
        initializer3 = RI.HeUniform(inmaps, outmaps, kernel, factor, rng=rng)
        params3 = initializer3(shape=(5, 5))

        assert not np.allclose(params1, params3)
        assert not np.allclose(params2, params3)

    def test_lecun_normal_unknown_mode(self):
        inmaps = 10
        outmaps = 10
        kernel = (10, 10)
        factor = 10.0

        with pytest.raises(NotImplementedError):
            RI.LeCunNormal(inmaps, outmaps, kernel, factor, mode="fan_out")

        with pytest.raises(NotImplementedError):
            RI.LeCunNormal(inmaps, outmaps, kernel, factor, mode="fan_unknown")

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3 * i, 5 * i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_normal_std_he_forward(self, inmap, outmap, kernel, factor):
        n = inmap * kernel[0] * kernel[1]
        expected = np.sqrt(factor / n)
        actual = RI.calc_normal_std_he_forward(inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3 * i, 5 * i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_normal_std_he_backward(self, inmap, outmap, kernel, factor):
        n = outmap * kernel[0] * kernel[1]
        expected = np.sqrt(factor / n)
        actual = RI.calc_normal_std_he_backward(inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3 * i, 5 * i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_uniform_lim_he_forward(self, inmap, outmap, kernel, factor):
        n = inmap * kernel[0] * kernel[1]
        expected = np.sqrt((3.0 * factor) / n)
        actual = RI.calc_uniform_lim_he_forward(inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize("inmap, outmap, kernel, factor", [(3 * i, 5 * i, (i, i), 0.5 * i) for i in range(1, 10)])
    def test_calc_uniform_lim_he_backward(self, inmap, outmap, kernel, factor):
        n = outmap * kernel[0] * kernel[1]
        expected = np.sqrt((3.0 * factor) / n)
        actual = RI.calc_uniform_lim_he_backward(inmaps=inmap, outmaps=outmap, kernel=kernel, factor=factor)
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
