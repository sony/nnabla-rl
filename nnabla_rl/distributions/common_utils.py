import numpy as np

import nnabla.functions as NF


def gaussian_log_prob(x, mean, var, ln_var):
    # log N(x|mu, var)
    # = -0.5*log2*pi - 0.5 * ln_var - 0.5 * (x-mu)**2 / var
    axis = len(x.shape) - 1
    return NF.sum(-0.5 * np.log(2.0 * np.pi) - 0.5 * ln_var - 0.5 * (x-mean)**2 / var, axis=axis, keepdims=True)
