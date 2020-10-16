import nnabla as nn
import nnabla.functions as F

import numpy as np


def compute_hessian(y, x):
    """ Compute hessian (= dy^2 / dx^2) in Naive way, 

    Args:
        y (nn.Variable): Outputs of the differentiable function.
        x (list[nn.Variable]): List of parameters
    Returns:
        hessian (numpy.ndarray): Hessian of outputs with respect to the parameters
    """
    for param in x:
        param.grad.zero()
    grads = nn.grad([y], x)
    if len(grads) > 1:
        flat_grads = F.concatenate(
            *[F.reshape(grad, (-1,), inplace=False) for grad in grads])
    else:
        flat_grads = F.reshape(grads[0], (-1,), inplace=False)
    flat_grads.need_grad = True

    hessian = np.zeros(
        (flat_grads.shape[0], flat_grads.shape[0]), dtype=np.float32)

    for i in range(flat_grads.shape[0]):
        flat_grads[i].forward()
        for param in x:
            param.grad.zero()
        flat_grads[i].backward()

        num_index = 0
        for param in x:
            grad = param.g.flatten()  # grad of grad so this is hessian
            hessian[i, num_index:num_index+len(grad)] = grad
            num_index += len(grad)

    return hessian
