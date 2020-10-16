import numpy as np

import nnabla as nn
import nnabla.functions as F


def conjugate_gradient(compute_Ax, b, max_iterations=10, residual_tol=1e-10):
    """ Conjugate gradient method to solve x = A^-1b
        We implemented that iteration version of conjugate gradient method
        This function minimize f(x) = 0.5 * x^TAx + bx

    Args:
        compute_Ax (callable funtion): function of computing Ax
        b (numpy.ndarray): vector, shape like as x
        max_iterations (int): number of maximum iteration, default is 10.
            If given None, iteration lasts until residual value is enough small
        residual_tol (float): residual value of tolerance.
    Returns:
        x (numpy.ndarray): optimization results, solving x = A^-1b
    """
    x = np.zeros_like(b)
    r = b - compute_Ax(x)
    p = r.copy()
    square_r = np.dot(r, r)
    iteration_number = 0

    while True:
        y = compute_Ax(p)
        alpha = square_r / np.dot(y, p)
        x = x + alpha * p
        r = r - alpha * y
        new_square_r = np.dot(r, r)

        if new_square_r < residual_tol:
            break

        if max_iterations is not None:
            if iteration_number >= max_iterations-1:
                break

        beta = new_square_r / square_r
        p = r + beta * p
        square_r = new_square_r
        iteration_number += 1

    return x
