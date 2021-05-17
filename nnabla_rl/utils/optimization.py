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


def conjugate_gradient(compute_Ax, b, max_iterations=10, residual_tol=1e-10):
    ''' Conjugate gradient method to solve x = A^-1b
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
    '''
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
