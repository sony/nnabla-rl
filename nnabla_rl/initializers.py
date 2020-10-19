import nnabla.initializer as I

import numpy as np


def HeNormal(inmaps, outmaps, kernel=(1, 1), factor=2.0, mode='fan_in'):
    """ Create Weight initialzier proposed by He et al. (Normal distribution version)

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
    """
    if mode == 'fan_in':
        s = calc_normal_std_he_forward(
            inmaps, outmaps, kernel, factor)
    elif mode == 'fan_out':
        s = calc_normal_std_he_backward(
            inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return I.NormalInitializer(s)


def LeCunNormal(inmaps, outmaps, kernel=(1, 1), factor=1.0, mode='fan_in'):
    """ Create Weight initialzier proposed in LeCun 98, Efficient Backprop (Normal distribution version)

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
    """
    if mode == 'fan_in':
        s = calc_normal_std_he_forward(inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return I.NormalInitializer(s)


def HeUniform(inmaps, outmaps, kernel=(1, 1), factor=2.0, mode='fan_in'):
    """ Create Weight initialzier proposed by He et al. (Uniform distribution version)

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
    """
    if mode == 'fan_in':
        lim = calc_uniform_lim_he_forward(inmaps, outmaps, kernel, factor)
    elif mode == 'fan_out':
        lim = calc_uniform_lim_he_backward(inmaps, outmaps, kernel, factor)
    else:
        raise NotImplementedError('Unknown init mode: {}'.format(mode))

    return I.UniformInitializer(lim=(-lim, lim))


class NormcInitializer(I.BaseInitializer):
    # See: https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
    def __init__(self, std=1.0, axis=0, rng=None):
        if rng is None:
            rng = I.random.prng
        self._rng = rng
        self._std = std
        self._axis = axis

    def __call__(self, shape):
        params = self._rng.randn(*shape)
        params *= self._std / \
            np.sqrt(np.square(params).sum(axis=self._axis, keepdims=True))
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
