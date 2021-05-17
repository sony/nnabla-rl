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

import nnabla as nn
import nnabla.functions as NF


def compute_hessian(y, x):
    ''' Compute hessian (= dy^2 / dx^2) in Naive way,

    Args:
        y (nn.Variable): Outputs of the differentiable function.
        x (list[nn.Variable]): List of parameters
    Returns:
        hessian (numpy.ndarray): Hessian of outputs with respect to the parameters
    '''
    for param in x:
        param.grad.zero()
    grads = nn.grad([y], x)
    if len(grads) > 1:
        flat_grads = NF.concatenate(
            *[NF.reshape(grad, (-1,), inplace=False) for grad in grads])
    else:
        flat_grads = NF.reshape(grads[0], (-1,), inplace=False)
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
