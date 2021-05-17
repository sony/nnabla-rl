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

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

import nnabla as nn
import nnabla.functions as NF


def sample_gaussian(mean: nn.Variable,
                    ln_var: nn.Variable,
                    noise_clip: Optional[Tuple[float, float]] = None) -> nn.Variable:
    '''
    Sample value from a gaussian distribution of given mean and variance.

    Args:
        mean (nn.Variable): Mean of the gaussian distribution
        ln_var (nn.Variable): Logarithm of the variance of the gaussian distribution
        noise_clip (Optional[Tuple(float, float)]): Clipping value of the sampled noise.

    Returns:
        nn.Variable: Sampled value from gaussian distribution of given mean and variance
    '''
    if not (mean.shape == ln_var.shape):
        raise ValueError('mean and ln_var has different shape')

    noise = NF.randn(shape=mean.shape)
    stddev = NF.exp(ln_var * 0.5)
    if noise_clip is not None:
        noise = NF.clip_by_value(noise, min=noise_clip[0], max=noise_clip[1])
    assert mean.shape == noise.shape
    return mean + stddev * noise


def sample_gaussian_multiple(mean: nn.Variable,
                             ln_var: nn.Variable,
                             num_samples: int,
                             noise_clip: Optional[Tuple[float, float]] = None) -> nn.Variable:
    '''
    Sample multiple values from a gaussian distribution of given mean and variance.
    The returned variable will have an additional axis in the middle as follows
    (batch_size, num_samples, dimension)

    Args:
        mean (nn.Variable): Mean of the gaussian distribution
        ln_var (nn.Variable): Logarithm of the variance of the gaussian distribution
        num_samples (int): Number of samples to sample
        noise_clip (Optional[Tuple(float, float)]): Clipping value of the sampled noise.

    Returns:
        nn.Variable: Sampled values from gaussian distribution of given mean and variance
    '''
    if not (mean.shape == ln_var.shape):
        raise ValueError('mean and ln_var has different shape')

    batch_size = mean.shape[0]
    data_shape = mean.shape[1:]
    mean = NF.reshape(mean, shape=(batch_size, 1, *data_shape))
    stddev = NF.reshape(NF.exp(ln_var * 0.5),
                        shape=(batch_size, 1, *data_shape))

    output_shape = (batch_size, num_samples, *data_shape)

    noise = NF.randn(shape=output_shape)
    if noise_clip is not None:
        noise = NF.clip_by_value(noise, min=noise_clip[0], max=noise_clip[1])
    sample = mean + stddev * noise
    assert sample.shape == output_shape
    return sample


def expand_dims(x: nn.Variable, axis: int) -> nn.Variable:
    '''
    Add dimension to target axis for the given variable

    Args:
        x (nn.Variable): Variable to expand the dimension
        axis (int): The axis to expand the dimension. Non negative.

    Returns:
        nn.Variable: Variable with additional dimension in the target axis
    '''
    target_shape = (*x.shape[0:axis], 1, *x.shape[axis:])
    return NF.reshape(x, shape=target_shape, inplace=False)


def repeat(x: nn.Variable, repeats: int, axis: int) -> nn.Variable:
    '''
    repeats the value along given axis for repeats times.

    Args:
        x (nn.Variable): Variable to repeat the values along given axis
        repeats (int): Number of times to repeat
        axis (int): The axis to expand the dimension. Non negative.

    Returns:
        nn.Variable: Variable with values repeated along given axis
    '''
    # TODO: Find more efficient way
    assert isinstance(repeats, int)
    assert axis is not None
    assert axis < len(x.shape)
    reshape_size = (*x.shape[0:axis+1], 1, *x.shape[axis+1:])
    repeater_size = (*x.shape[0:axis+1], repeats, *x.shape[axis+1:])
    final_size = (*x.shape[0:axis], x.shape[axis] * repeats, *x.shape[axis+1:])
    x = NF.reshape(x=x, shape=reshape_size)
    x = NF.broadcast(x, repeater_size)
    return NF.reshape(x, final_size)


def sqrt(x: nn.Variable):
    '''
    Compute the squared root of given variable

    Args:
        x (nn.Variable): Variable to compute the squared root

    Returns:
        nn.Variable: Squared root of given variable
    '''
    return NF.pow_scalar(x, 0.5)


def std(x: nn.Variable, axis: Optional[int] = None, keepdims: bool = False) -> nn.Variable:
    '''
    Compute the standard deviation of given variable along axis.

    Args:
        x (nn.Variable): Variable to compute the squared root
        axis (Optional[int]): Axis to compute the standard deviation. Defaults to None. None will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axis are kept as a dimension with 1 element.

    Returns:
        nn.Variable: Standard deviation of given variable along axis.
    '''
    # sigma = sqrt(E[(X - E[X])^2])
    mean = NF.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    variance = NF.mean(diff**2, axis=axis, keepdims=keepdims)
    return sqrt(variance)


def argmax(x: nn.Variable, axis: Optional[int] = None, keepdims: bool = False) -> nn.Variable:
    '''
    Compute the index which given variable has maximum value along the axis.

    Args:
        x (nn.Variable): Variable to compute the argmax
        axis (Optional[int]): Axis to compare the values. Defaults to None. None will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axis are kept as a dimension with 1 element.

    Returns:
        nn.Variable: Index of the variable which its value is maximum along the axis
    '''
    return NF.max(x=x, axis=axis, keepdims=keepdims, with_index=True, only_index=True)


def quantile_huber_loss(x0: nn.Variable, x1: nn.Variable, kappa: float, tau: nn.Variable) -> nn.Variable:
    '''
    Compute the quantile huber loss
    See following papers for details:
    https://arxiv.org/pdf/1710.10044.pdf
    https://arxiv.org/pdf/1806.06923.pdf

    Args:
        x0 (nn.Variable): Quantile values
        x1 (nn.Variable): Quantile values
        kappa (float): Threshold value of huber loss which switches the loss value between squared loss and linear loss
        tau (nn.Variable): Quantile targets

    Returns:
        nn.Variable: Quantile huber loss
    '''
    u = x0 - x1
    # delta(u < 0)
    delta = NF.less_scalar(u, val=0.0)
    delta.need_grad = False
    assert delta.shape == u.shape

    if kappa <= 0.0:
        return u * (tau - delta)
    else:
        Lk = NF.huber_loss(x0, x1, delta=kappa) * 0.5
        assert Lk.shape == u.shape
        return NF.abs(tau - delta) * Lk / kappa


def mean_squared_error(x0: nn.Variable, x1: nn.Variable) -> nn.Variable:
    '''
    Convenient alias for mean squared error operation

    Args:
        x0 (nn.Variable): N-D array
        x1 (nn.Variable): N-D array

    Returns:
        nn.Variable: Mean squared error between x0 and x1
    '''
    return NF.mean(NF.squared_error(x0, x1))


def minimum_n(variables: Sequence[nn.Variable]) -> nn.Variable:
    '''
    Compute the minimum among the list of variables

    Args:
        variables (Sequence[nn.Variable]): Sequence of variables. All the variables must have same shape.

    Returns:
        nn.Variable: Minimum value among the list of variables
    '''
    if len(variables) < 1:
        raise ValueError('Variables must have at least 1 variable')
    if len(variables) == 1:
        return variables[0]
    if len(variables) == 2:
        return NF.minimum2(variables[0], variables[1])

    minimum = NF.minimum2(variables[0], variables[1])
    for variable in variables[2:]:
        minimum = NF.minimum2(minimum, variable)
    return minimum


def gaussian_cross_entropy_method(objective_function: Callable[[nn.Variable], nn.Variable],
                                  init_mean: nn.Variable, init_var: nn.Variable,
                                  pop_size: int = 500, num_elites: int = 10,
                                  num_iterations: int = 5, alpha: float = 0.25) -> Tuple[nn.Variable, nn.Variable]:
    ''' Cross Entropy Method using gaussian distribution.
        This function optimized objective function J(x), where x is variable.

    Examples:
        >>> import nnabla as nn
        >>> import nnabla.functions as NF
        >>> import numpy as np
        >>> import nnabla_rl.functions as RF

        >>> def objective_function(x): return -((x - 3.)**2)

        >>> batch_size = 1
        >>> variable_size = 1

        >>> init_mean = nn.Variable.from_numpy_array(np.zeros((batch_size, state_size)))
        >>> init_var = nn.Variable.from_numpy_array(np.ones((batch_size, state_size)))
        >>> optimal_x, _ = RF.gaussian_cross_entropy_method(objective_function, init_mean, init_var, alpha=0)

        >>> optimal_x.forward()
        >>> optimal_x.shape
        (1, 1)  # (batch_size, variable_size)
        >>> optimal_x.d
        array([[3.]], dtype=float32)

    Args:
        objective_function (Callable[[nn.Variable], nn.Variable]): objective function
        init_mean (nn.Variable): initial mean
        init_var (nn.Variable): initial variance
        pop_size (int): pop size
        num_elites (int): number of elites
        num_iterations (int): number of iterations
        alpha (float): parameter of soft update

    Returns:
        Tuple[nn.Variable, nn.Variable]: mean of elites samples and top of elites samples
    '''
    mean = init_mean
    var = init_var
    batch_size, gaussian_dimension = mean.shape

    elite_arange_index = np.tile(np.arange(batch_size)[:, np.newaxis], (1, num_elites))[np.newaxis, :, :]
    elite_arange_index = nn.Variable.from_numpy_array(elite_arange_index)
    top_arange_index = np.tile(np.arange(batch_size)[:, np.newaxis], (1, 1))[np.newaxis, :, :]
    top_arange_index = nn.Variable.from_numpy_array(top_arange_index)

    for _ in range(num_iterations):
        # samples.shape = (batch_size, pop_size, gaussian_dimension)
        samples = sample_gaussian_multiple(mean, NF.log(var), pop_size)
        # values.shape = (batch_size*pop_size, 1)
        values = objective_function(samples.reshape((-1, gaussian_dimension)))
        values = values.reshape((batch_size, pop_size, 1))

        elites_index = NF.sort(values, axis=1, reverse=True, with_index=True, only_index=True)[:, :num_elites, :]
        elites_index = elites_index.reshape((1, batch_size, num_elites))
        elites_index = NF.concatenate(elite_arange_index, elites_index, axis=0)

        top_index = NF.max(values, axis=1, with_index=True, only_index=True, keepdims=True)
        top_index = top_index.reshape((1, batch_size, 1))
        top_index = NF.concatenate(top_arange_index, top_index, axis=0)

        # elite.shape = (batch_size, num_elites, gaussian_dimension)
        elites = NF.gather_nd(samples, elites_index)
        # top.shape = (batch_size, gaussian_dimension)
        top = NF.gather_nd(samples, top_index).reshape((batch_size, gaussian_dimension))

        # new_mean.shape = (batch_size, 1, gaussian_dimension)
        new_mean = NF.mean(elites, axis=1, keepdims=True)
        # new_var.shape = (batch_size, 1, gaussian_dimension)
        new_var = NF.mean((elites - new_mean)**2, axis=1, keepdims=True)

        mean = alpha * mean + (1 - alpha) * new_mean.reshape((batch_size, gaussian_dimension))
        var = alpha * var + (1 - alpha) * new_var.reshape((batch_size, gaussian_dimension))

    return mean, top
