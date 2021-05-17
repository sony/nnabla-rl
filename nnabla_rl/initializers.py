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

import nnabla.initializer as NI


def HeNormal(inmaps, outmaps, kernel=(1, 1), factor=2.0, mode='fan_in'):
    ''' Create Weight initializer proposed by He et al. (Normal distribution version)

    Args:
        inmaps (int): Map size of an input Variable,
        outmaps (int): Map size of an output Variable,
        kernel (tuple(int) or None): Convolution kernel spatial shape.
            In Affine, use the default setting
        factor (float): Coefficient applied to the standard deviation computation. default is 2.0
        mode (str): 'fan_in' or 'fan_out' is supported.
    Returns:
        HeNormal : weight initialzier
    Raises:
        NotImplementedError: mode other than 'fan_in' or 'fan_out' is given
    '''
    if mode == 'fan_in':
        s = calc_normal_std_he_forward(
            inmaps, outmaps, kernel, factor)
    elif mode == 'fan_out':
        s = calc_normal_std_he_backward(
            inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return NI.NormalInitializer(s)


def LeCunNormal(inmaps, outmaps, kernel=(1, 1), factor=1.0, mode='fan_in'):
    ''' Create Weight initializer proposed in LeCun 98, Efficient Backprop (Normal distribution version)

    Args:
        inmaps (int): Map size of an input Variable,
        outmaps (int): Map size of an output Variable,
        kernel (tuple(int) or None): Convolution kernel spatial shape.
            In Affine, use the default setting
        factor (float): Coefficient applied to the standard deviation computation. default is 1.0
        mode (str): 'fan_in' is the only mode supported for this initializer.
    Returns:
        LeCunNormal : weight initialzier
    Raises:
        NotImplementedError: mode other than 'fan_in' is given
    '''
    if mode == 'fan_in':
        s = calc_normal_std_he_forward(inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return NI.NormalInitializer(s)


def HeUniform(inmaps, outmaps, kernel=(1, 1), factor=2.0, mode='fan_in'):
    ''' Create Weight initializer proposed by He et al. (Uniform distribution version)

    Args:
        inmaps (int): Map size of an input Variable,
        outmaps (int): Map size of an output Variable,
        kernel (tuple(int) or None): Convolution kernel spatial shape.
            In Affine, use the default setting
        factor (float): Coefficient applied to the uniform distribution limit computation. default is 2.0
        mode (str): 'fan_in' or 'fan_out' is supported.
    Returns:
        HeUniform : weight initialzier
    Raises:
        NotImplementedError: mode other than 'fan_in' or 'fan_out' is given
    '''
    if mode == 'fan_in':
        lim = calc_uniform_lim_he_forward(inmaps, outmaps, kernel, factor)
    elif mode == 'fan_out':
        lim = calc_uniform_lim_he_backward(inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return NI.UniformInitializer(lim=(-lim, lim))


def GlorotUniform(inmaps, outmaps, kernel=(1, 1)):
    lb, ub = NI.calc_uniform_lim_glorot(inmaps, outmaps, kernel)
    return NI.UniformInitializer(lim=(lb, ub))


class NormcInitializer(NI.BaseInitializer):
    ''' Create Normc initializer
    See: https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
    Initializes the parameter which normalized along 'axis' dimension.

    Args:
        std (float): normalization scaling value. Defaults to 1.
        axis (int): dimension to normalize. Defaults to 0.
        rng (np.random.RandomState):
            Random number generator to sample numbers from. Defaults to None.
            When None, NNabla's default random nunmber generator will be used.
    Returns:
        NormcInitializer : weight initialzier
    '''

    def __init__(self, std=1.0, axis=0, rng=None):
        if rng is None:
            rng = NI.random.prng
        self._rng = rng
        self._std = std
        self._axis = axis

    def __call__(self, shape):
        params = self._rng.randn(*shape)
        params *= self._std / np.sqrt(np.square(params).sum(axis=self._axis, keepdims=True))
        return params


def calc_normal_std_he_forward(inmaps, outmaps, kernel, factor):
    n = inmaps * np.prod(kernel)
    return np.sqrt(factor / n)


def calc_normal_std_he_backward(inmaps, outmaps, kernel, factor):
    n = outmaps * np.prod(kernel)
    return np.sqrt(factor / n)


def calc_uniform_lim_he_forward(inmaps, outmaps, kernel, factor):
    n = inmaps * np.prod(kernel)
    return np.sqrt(3.0 * factor / n)


def calc_uniform_lim_he_backward(inmaps, outmaps, kernel, factor):
    n = outmaps * np.prod(kernel)
    return np.sqrt(3.0 * factor / n)
