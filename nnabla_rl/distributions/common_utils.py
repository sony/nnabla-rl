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

import nnabla.functions as NF


def gaussian_log_prob(x, mean, var, ln_var):
    # log N(x|mu, var)
    # = -0.5*log2*pi - 0.5 * ln_var - 0.5 * (x-mu)**2 / var
    axis = len(x.shape) - 1
    return NF.sum(-0.5 * np.log(2.0 * np.pi) - 0.5 * ln_var - 0.5 * (x-mean)**2 / var, axis=axis, keepdims=True)
