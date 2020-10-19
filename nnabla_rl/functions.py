import nnabla as nn
import nnabla.functions as F


def sample_gaussian(mean, ln_var, noise_clip=None):
    assert isinstance(mean, nn.Variable)
    assert isinstance(ln_var, nn.Variable)
    if not (mean.shape == ln_var.shape):
        raise ValueError('mean and ln_var has different shape')

    noise = F.randn(shape=mean.shape)
    stddev = F.exp(ln_var * 0.5)
    if noise_clip is not None:
        noise = F.clip_by_value(noise, min=noise_clip[0], max=noise_clip[1])
    assert mean.shape == noise.shape
    return mean + stddev * noise


def sample_gaussian_multiple(mean, ln_var, num_samples, noise_clip=None):
    assert isinstance(mean, nn.Variable)
    assert isinstance(ln_var, nn.Variable)
    if not (mean.shape == ln_var.shape):
        raise ValueError('mean and ln_var has different shape')

    batch_size = mean.shape[0]
    data_shape = mean.shape[1:]
    mean = F.reshape(mean, shape=(batch_size, 1, *data_shape))
    stddev = F.reshape(F.exp(ln_var * 0.5), shape=(batch_size, 1, *data_shape))

    output_shape = (batch_size, num_samples, *data_shape)

    noise = F.randn(shape=output_shape)
    if noise_clip is not None:
        noise = F.clip_by_value(noise, min=noise_clip[0], max=noise_clip[1])
    sample = mean + stddev * noise
    assert sample.shape == output_shape
    return sample


def expand_dims(x, axis):
    target_shape = (*x.shape[0:axis], 1, *x.shape[axis:])
    return F.reshape(x, shape=target_shape, inplace=False)


def repeat(x, repeats, axis):
    # TODO: Find more efficient way
    assert isinstance(repeats, int)
    assert axis is not None
    assert axis < len(x.shape)
    reshape_size = (*x.shape[0:axis+1], 1, *x.shape[axis+1:])
    repeater_size = (*x.shape[0:axis+1], repeats, *x.shape[axis+1:])
    final_size = (*x.shape[0:axis], x.shape[axis] * repeats, *x.shape[axis+1:])
    x = F.reshape(x=x, shape=reshape_size)
    x = F.broadcast(x, repeater_size)
    return F.reshape(x, final_size)


def sqrt(x):
    return F.pow_scalar(x, 0.5)


def std(x, axis=None, keepdims=False):
    # sigma = sqrt(E[(X - E[X])^2])
    mean = F.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    variance = F.mean(diff**2, axis=axis, keepdims=keepdims)
    return sqrt(variance)


def argmax(x, axis=None):
    return F.max(x=x, axis=axis, with_index=True, only_index=True)


def quantile_huber_loss(x0, x1, kappa, tau):
    ''' Quantile huber loss
    See following papers for details:
    https://arxiv.org/pdf/1710.10044.pdf
    https://arxiv.org/pdf/1806.06923.pdf
    '''
    u = x0 - x1
    # delta(u < 0)
    delta = F.less_scalar(u, val=0.0)
    delta.need_grad = False
    assert delta.shape == u.shape

    if kappa <= 0.0:
        return u * (tau - delta)
    else:
        Lk = F.huber_loss(x0, x1, delta=kappa) * 0.5
        assert Lk.shape == u.shape
        return F.abs(tau - delta) * Lk / kappa


def mean_squared_error(x0, x1):
    return F.mean(F.squared_error(x0, x1))
