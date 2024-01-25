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

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.functions as RF
import nnabla_rl.random as rl_random


class TestFunctions(object):
    def test_sample_gaussian(self):
        batch_size = 1000
        output_dim = 5

        input_shape = (batch_size, output_dim)
        mean = np.ones(shape=input_shape) * 5
        sigma = np.ones(shape=input_shape) * 100
        ln_sigma = np.log(sigma)

        mean_var = nn.Variable(input_shape)
        mean_var.d = mean

        ln_sigma_var = nn.Variable(input_shape)
        ln_sigma_var.d = ln_sigma

        sampled_value = RF.sample_gaussian(
            mean=mean_var, ln_var=(ln_sigma_var * 2.0))
        assert sampled_value.shape == (batch_size, output_dim)

    def test_sample_gaussian_wrong_parameter_shape(self):
        batch_size = 10
        mean_dim = 5
        sigma_dim = 10

        mean = np.ones(shape=(batch_size, mean_dim)) * 5
        sigma = np.ones(shape=(batch_size, sigma_dim)) * 100
        ln_sigma = np.log(sigma)
        mean_var = nn.Variable(mean.shape)
        mean_var.d = mean

        ln_sigma_var = nn.Variable(sigma.shape)
        ln_sigma_var.d = ln_sigma

        with pytest.raises(ValueError):
            RF.sample_gaussian(
                mean=mean_var, ln_var=(ln_sigma_var * 2.0))

    def test_expand_dims(self):
        batch_size = 4
        data_h = 3
        data_w = 2

        data = np.random.normal(size=(batch_size, data_h, data_w))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        actual = RF.expand_dims(data_var, axis=0)
        actual.forward()
        expected = np.expand_dims(data, axis=0)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.expand_dims(data_var, axis=1)
        actual.forward()
        expected = np.expand_dims(data, axis=1)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.expand_dims(data_var, axis=2)
        actual.forward()
        expected = np.expand_dims(data, axis=2)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

    def test_repeat(self):
        batch_size = 4
        data_h = 3
        data_w = 2

        data = np.random.normal(size=(batch_size, data_h, data_w))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        repeats = 5

        actual = RF.repeat(data_var, repeats=repeats, axis=0)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=0)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.repeat(data_var, repeats=repeats, axis=1)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=1)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

        actual = RF.repeat(data_var, repeats=repeats, axis=2)
        actual.forward()
        expected = np.repeat(data, repeats=repeats, axis=2)
        assert actual.shape == expected.shape
        assert np.alltrue(actual.data.data == expected)

    def test_sqrt(self):
        num_samples = 100
        batch_num = 100
        # exp to enforce positive value
        data = np.exp(np.random.normal(size=(num_samples,  batch_num, 1)))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        actual = RF.sqrt(data_var)
        actual.forward(clear_buffer=True)
        expected = np.sqrt(data)
        assert actual.shape == expected.shape
        assert np.all(np.isclose(actual.data.data, expected))

    def test_std(self):
        # stddev computation
        num_samples = 100
        batch_num = 100
        data = np.random.normal(size=(num_samples,  batch_num, 1))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        for axis in range(len(data.shape)):
            actual = RF.std(data_var, axis=axis, keepdims=False)
            actual.forward(clear_buffer=True)
            expected = np.std(data, axis=axis, keepdims=False)
            assert actual.shape == expected.shape
            assert np.all(np.isclose(actual.data.data, expected))

            actual = RF.std(data_var, axis=axis, keepdims=True)
            actual.forward(clear_buffer=True)
            expected = np.std(data, axis=axis, keepdims=True)
            assert actual.shape == expected.shape
            assert np.all(np.isclose(actual.data.data, expected))

    def test_argmax(self):
        num_samples = 100
        batch_num = 100
        data = np.random.normal(size=(num_samples,  batch_num, 1))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        for axis in [None, *range(len(data.shape))]:
            actual = RF.argmax(data_var, axis=axis)
            actual.forward(clear_buffer=True)
            expected = np.argmax(data, axis=axis)
            assert actual.shape == expected.shape
            assert np.all(actual.data.data == expected)

    def test_argmin(self):
        num_samples = 100
        batch_num = 100
        data = np.random.normal(size=(num_samples,  batch_num, 1))
        data = np.float32(data)
        data_var = nn.Variable(data.shape)
        data_var.d = data

        for axis in [None, *range(len(data.shape))]:
            actual = RF.argmin(data_var, axis=axis)
            actual.forward(clear_buffer=True)
            expected1 = np.argmin(data, axis=axis)
            expected2 = RF.argmax(-data_var, axis=axis)
            expected2.forward()
            assert actual.shape == expected1.shape
            assert actual.shape == expected2.shape
            assert np.all(actual.data.data == expected1)
            assert np.all(actual.data.data == expected2.data.data)

    def test_quantile_huber_loss(self):
        def huber_loss(x0, x1, kappa):
            diff = x0 - x1
            flag = (np.abs(diff) < kappa).astype(np.float32)
            return (flag) * 0.5 * (diff ** 2.0) + (1.0 - flag) * kappa * (np.abs(diff) - 0.5 * kappa)

        def quantile_huber_loss(x0, x1, kappa, tau):
            u = x0 - x1
            delta = np.less(u, np.zeros(shape=u.shape, dtype=np.float32))
            if kappa == 0.0:
                return (tau - delta) * u
            else:
                Lk = huber_loss(x0, x1, kappa=kappa)
                return np.abs(tau - delta) * Lk / kappa

        N = 10
        batch_size = 1
        x0 = np.random.normal(size=(batch_size, N))
        x1 = np.random.normal(size=(batch_size, N))
        for kappa in [0.0, 1.0]:
            tau = np.array([i / N for i in range(1, N + 1)]).reshape((1, -1))
            tau = np.repeat(tau, axis=0, repeats=batch_size)
            tau_var = nn.Variable.from_numpy_array(tau)
            x0_var = nn.Variable.from_numpy_array(x0)
            x1_var = nn.Variable.from_numpy_array(x1)
            loss = RF.quantile_huber_loss(x0=x0_var, x1=x1_var, kappa=kappa, tau=tau_var)
            loss.forward()

            actual = loss.d
            expected = quantile_huber_loss(x0=x0, x1=x1, kappa=kappa, tau=tau)

            assert actual.shape == expected.shape
            assert np.allclose(actual, expected, atol=1e-7)

    def test_mean_squared_error(self):
        num_samples = 100
        batch_num = 100
        # exp to enforce positive value
        x0 = np.exp(np.random.normal(size=(batch_num, num_samples)))
        x0_var = nn.Variable.from_numpy_array(x0)

        x1 = np.exp(np.random.normal(size=(batch_num, num_samples)))
        x1_var = nn.Variable.from_numpy_array(x1)

        actual = RF.mean_squared_error(x0_var, x1_var)
        actual.forward(clear_buffer=True)
        expected = np.mean((x0 - x1)**2)
        assert actual.shape == expected.shape
        assert np.all(np.isclose(actual.d, expected))

    def test_gaussian_cross_entropy_method(self):
        def objective_function(x):
            return -((x - 3.)**2)

        batch_size = 1
        var_size = 1

        init_mean = nn.Variable.from_numpy_array(np.zeros((batch_size, var_size)))
        init_var = nn.Variable.from_numpy_array(np.ones((batch_size, var_size))*4.)
        optimal_mean, optimal_top = RF.gaussian_cross_entropy_method(
            objective_function, init_mean, init_var, alpha=0., num_iterations=10)

        nn.forward_all([optimal_mean, optimal_top])

        assert np.allclose(optimal_mean.d, np.array([[3.]]), atol=1e-2)
        assert np.allclose(optimal_top.d, np.array([[3.]]), atol=1e-2)

    def test_gaussian_cross_entropy_method_with_complicated_objective_function(self):
        def dummy_q_function(s, a):
            return -((a - s)**2)

        batch_size = 5
        pop_size = 500
        state_size = 1
        action_size = 1

        s = np.arange(batch_size*state_size).reshape(batch_size, state_size)
        s = np.tile(s, (pop_size, 1, 1))
        s = np.transpose(s, (1, 0, 2))
        s_var = nn.Variable.from_numpy_array(s.reshape(batch_size, pop_size, state_size))
        def objective_function(x): return dummy_q_function(s_var, x)

        init_mean = nn.Variable.from_numpy_array(np.zeros((batch_size, action_size)))
        init_var = nn.Variable.from_numpy_array(np.ones((batch_size, action_size))*4)
        optimal_mean, optimal_top = RF.gaussian_cross_entropy_method(
            objective_function, init_mean, init_var, pop_size, alpha=0., num_iterations=10)

        nn.forward_all([optimal_mean, optimal_top])

        assert np.allclose(optimal_mean.d, np.array([[0.], [1.], [2.], [3.], [4.]]), atol=1e-2)
        assert np.allclose(optimal_top.d, np.array([[0.], [1.], [2.], [3.], [4.]]), atol=1e-2)

    def test_random_shooting_method(self):
        def objective_function(x):
            return -((x - 3.)**2)

        batch_size = 1
        var_size = 1

        upper_bound = np.ones((batch_size, var_size)) * 3.5
        lower_bound = np.ones((batch_size, var_size)) * 2.5
        optimal_top = RF.random_shooting_method(
            objective_function, upper_bound, lower_bound)

        nn.forward_all([optimal_top])

        assert np.allclose(optimal_top.d, np.array([[3.]]), atol=1e-1)

    def test_random_shooting_method_with_complicated_objective_function(self):
        def dummy_q_function(s, a):
            return -((a - s)**2)

        batch_size = 5
        sample_size = 500
        state_size = 1
        action_size = 1

        s = np.arange(batch_size*state_size).reshape(batch_size, state_size)
        s = np.tile(s, (sample_size, 1, 1))
        s = np.transpose(s, (1, 0, 2))
        s_var = nn.Variable.from_numpy_array(s.reshape(batch_size, sample_size, state_size))
        def objective_function(x): return dummy_q_function(s_var, x)

        upper_bound = np.ones((batch_size, action_size))*5
        lower_bound = np.zeros((batch_size, action_size))
        optimal_top = RF.random_shooting_method(objective_function, upper_bound, lower_bound)

        nn.forward_all([optimal_top])

        assert np.allclose(optimal_top.d, np.array([[0.], [1.], [2.], [3.], [4.]]), atol=1e-1)

    def test_random_shooting_method_with_invalid_bounds(self):
        def objective_function(x):
            return -((x - 3.)**2)

        batch_size = 1
        var_size = 1

        upper_bound = -np.ones((batch_size, var_size)) * 3.5
        lower_bound = np.ones((batch_size, var_size)) * 2.5

        with pytest.raises(ValueError):
            RF.random_shooting_method(objective_function, upper_bound, lower_bound)

    @pytest.mark.parametrize("batch_size", [i for i in range(1, 3)])
    @pytest.mark.parametrize("diag_size", [i for i in range(1, 5)])
    @pytest.mark.parametrize("upper", [True, False])
    def test_triangular_matrix(self, batch_size, diag_size, upper):
        non_diag_size = diag_size * (diag_size - 1) // 2
        diagonal = nn.Variable.from_numpy_array(np.random.normal(size=(batch_size, diag_size)).astype(np.float32))
        non_diagonal = nn.Variable.from_numpy_array(np.random.normal(
            size=(batch_size, non_diag_size)).astype(np.float32))

        triangular_matrix = RF.triangular_matrix(diagonal, non_diagonal, upper)
        triangular_matrix.forward()

        for b in range(batch_size):
            i = 0
            for row in range(diag_size):
                for col in range(diag_size):
                    value = triangular_matrix.d[b, row, col]
                    if col < row:
                        if upper:
                            assert value == 0
                        else:
                            assert value == non_diagonal.d[b, i]
                            i += 1
                    elif row < col:
                        if upper:
                            assert value == non_diagonal.d[b, i]
                            i += 1
                        else:
                            assert value == 0
                    else:
                        assert row == col
                        assert value == diagonal.d[b, row]

    @pytest.mark.parametrize("batch_size", [i for i in range(1, 3)])
    @pytest.mark.parametrize("diag_size", [i for i in range(1, 5)])
    @pytest.mark.parametrize("upper", [True, False])
    def test_triangular_matrix_create_diagonal_matrix(self, batch_size, diag_size, upper):
        diagonal = nn.Variable.from_numpy_array(np.random.normal(size=(batch_size, diag_size)).astype(np.float32))
        non_diagonal = None

        triangular_matrix = RF.triangular_matrix(diagonal, non_diagonal, upper)
        triangular_matrix.forward()

        for b in range(batch_size):
            for row in range(diag_size):
                for col in range(diag_size):
                    value = triangular_matrix.d[b, row, col]
                    if row == col:
                        assert value == diagonal.d[b, row]
                    else:
                        assert value == 0

    @pytest.mark.parametrize("batch_size", [i for i in range(1, 4)])
    @pytest.mark.parametrize("shape", [(1, ), (1, 2), (2, 3), (4, 5, 6)])
    def test_batch_flatten(self, batch_size, shape):
        x = nn.Variable.from_numpy_array(np.random.normal(size=(batch_size, *shape)).astype(np.float32))
        flattened_x = RF.batch_flatten(x)

        assert np.prod(x.shape) == np.prod(flattened_x.shape)
        assert batch_size == flattened_x.shape[0]
        assert len(flattened_x.shape) == 2

    def test_stop_gradient(self):
        import nnabla.functions as NF
        import nnabla.parametric_functions as NPF
        import nnabla_rl.distributions as D
        from nnabla_rl.models import StochasticPolicy

        class TestModel(StochasticPolicy):
            def pi(self, s: nn.Variable):
                with nn.parameter_scope(self.scope_name):
                    z = NPF.affine(s, n_outmaps=5)
                return D.Gaussian(z, nn.Variable.from_numpy_array(np.zeros(z.shape)))

        test_model = TestModel('test')
        for parameter in test_model.get_parameters().values():
            parameter.grad.zero()

        batch_size = 3
        shape = (10, )
        s = nn.Variable.from_numpy_array(np.random.normal(size=(batch_size, *shape)).astype(np.float32))
        distribution = test_model.pi(s)

        h = distribution.mean()
        h_sg = RF.stop_gradient(h)

        out = NF.mean(h * 5)
        out_sg = NF.mean(h_sg * 5)

        nn.forward_all((out, out_sg))

        np.testing.assert_allclose(out.d, out_sg.d)
        for parameter in test_model.get_parameters().values():
            assert np.allclose(parameter.g, 0)

        # should not backpropagate
        out_sg.backward()
        for parameter in test_model.get_parameters().values():
            assert np.allclose(parameter.g, 0)

        # this should backpropagate
        out.backward()
        for parameter in test_model.get_parameters().values():
            assert not np.allclose(parameter.g, 0)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("middle_size", [1, 2, 3])
    @pytest.mark.parametrize("data_size", [1, 2, 3])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_pytorch_equivalent_gather(self, batch_size, middle_size, data_size, axis):
        indices_shape = [batch_size, middle_size, data_size]
        data_shape = np.copy(indices_shape)
        data_shape[axis] *= 2

        data = np.random.normal(size=data_shape)
        low = 0
        high = data_shape[axis]
        indices = rl_random.drng.integers(low, high, size=indices_shape)

        data_var = nn.Variable.from_numpy_array(data)
        indices_var = nn.Variable.from_numpy_array(indices)
        actual = RF.pytorch_equivalent_gather(data_var, indices_var, axis=axis)
        actual.forward()
        expected = np.take_along_axis(data, indices, axis=axis)

        assert actual.shape == expected.shape
        assert np.allclose(actual.d, expected)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("middle_size", [1, 2, 3])
    @pytest.mark.parametrize("data_size", [1, 2, 3])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_concat_interleave(self, batch_size, middle_size, data_size, axis):
        data1 = np.random.normal(size=(batch_size, middle_size, data_size))
        data2 = np.random.normal(size=(batch_size, middle_size, data_size))
        data3 = np.random.normal(size=(batch_size, middle_size, data_size))

        var1 = nn.Variable.from_numpy_array(data1)
        var2 = nn.Variable.from_numpy_array(data2)
        var3 = nn.Variable.from_numpy_array(data3)

        interleaved = RF.concat_interleave((var1, var2, var3), axis=axis)
        interleaved.forward()

        interleaved = np.asarray(interleaved.d)

        assert interleaved.shape[axis] == data1.shape[axis] * 3

        if axis == 0:
            assert np.allclose(interleaved[0::3, ...], data1)
            assert np.allclose(interleaved[1::3, ...], data2)
            assert np.allclose(interleaved[2::3, ...], data3)
        if axis == 1:
            assert np.allclose(interleaved[..., 0::3, :], data1)
            assert np.allclose(interleaved[..., 1::3, :], data2)
            assert np.allclose(interleaved[..., 2::3, :], data3)
        if axis == 2:
            assert np.allclose(interleaved[..., 0::3], data1)
            assert np.allclose(interleaved[..., 1::3], data2)
            assert np.allclose(interleaved[..., 2::3], data3)

    @pytest.mark.parametrize("axis1", [0, 1, 2])
    @pytest.mark.parametrize("axis2", [0, 1, 2])
    def test_swapaxes(self, axis1, axis2):
        original = nn.Variable.from_numpy_array(np.empty(shape=(3, 4, 5)))
        swapped = RF.swapaxes(original, axis1, axis2)
        swapped.forward()

        for i in range(len(original.shape)):
            if i == axis1:
                assert swapped.shape[axis2] == original.shape[i]
            elif i == axis2:
                assert swapped.shape[axis1] == original.shape[i]
            else:
                assert swapped.shape[i] == original.shape[i]

        reswapped = RF.swapaxes(swapped, axis1, axis2)
        reswapped.forward()

        assert np.allclose(original.d, reswapped.d)

    @pytest.mark.parametrize("x, mean, std, value_clip, expected",
                             [
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]), None,  np.array([5.0])),
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]),  (-1.5, 1.5), np.array([1.5])),
                                 (np.array([-2.0]), np.array([1.0]), np.array([0.2]),  (-1.5, 1.5), np.array([-1.5])),
                                 (np.array([[2.0], [1.0]]), np.array([[1.0]]), np.array([[0.2]]), None,
                                  np.array([[5.0], [0.0]])),
                             ])
    def test_normalize(self, x, expected, mean, std, value_clip):
        x_var = nn.Variable.from_numpy_array(x)
        mean_var = nn.Variable.from_numpy_array(mean)
        std_var = nn.Variable.from_numpy_array(std)

        actual_var = RF.normalize(x_var, mean_var, std_var, value_clip=value_clip)
        actual_var.forward()

        assert np.allclose(actual_var.d, expected)

    @pytest.mark.parametrize("x, mean, std, value_clip, expected",
                             [
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]), None, np.array([1.4])),
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]),  (-1.0, 1.0), np.array([1.0])),
                                 (np.array([-2.0]), np.array([-1.0]), np.array([0.2]),  (-1.0, 1.0),  np.array([-1.0])),
                                 (np.array([[2.0], [1.0]]), np.array([[1.0]]), np.array([[0.2]]), None,
                                  np.array([[1.4], [1.2]])),
                             ])
    def test_unnormalize(self, x, expected, mean, std, value_clip):
        x_var = nn.Variable.from_numpy_array(x)
        mean_var = nn.Variable.from_numpy_array(mean)
        std_var = nn.Variable.from_numpy_array(std)

        actual_var = RF.unnormalize(x_var, mean_var, std_var, value_clip=value_clip)
        actual_var.forward()

        assert np.allclose(actual_var.d, expected)

    @pytest.mark.parametrize("var, epsilon, mode_for_floating_point_error, expected",
                             [
                                 (np.array([3.0]), 1.0, "add", np.array([2.0])),
                                 (np.array([4.0]), 0.01, "max", np.array([2.0])),
                                 (np.array([0.4]), 1.0, "max", np.array([1.0])),
                                 (np.array([[3.0], [8.0]]), 1.0, "add", np.array([[2.0], [3.0]])),
                                 (np.array([[4.0], [9.0]]), 0.01, "max", np.array([[2.0], [3.0]])),
                                 (np.array([[0.4], [0.9]]), 1.0, "max", np.array([[1.0], [1.0]])),
                             ])
    def test_compute_std(self, var, epsilon, mode_for_floating_point_error, expected):
        variance_variable = nn.Variable.from_numpy_array(var)
        actual_var = RF.compute_std(variance_variable, epsilon, mode_for_floating_point_error)
        actual_var.forward()

        assert np.allclose(actual_var.d, expected)

    def test_compute_std_with_invalid_args(self):
        variance_variable = nn.Variable.from_numpy_array(np.ones(1))
        with pytest.raises(ValueError):
            RF.compute_std(variance_variable, 0.01, "dummy_add")


if __name__ == "__main__":
    pytest.main()
