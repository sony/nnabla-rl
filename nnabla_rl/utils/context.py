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

import warnings
from typing import Any, Dict

import nnabla as nn
from nnabla.ext_utils import get_extension_context

contexts: Dict[int, Any] = {}


def set_nnabla_context(gpu_id):
    ctx = get_nnabla_context(gpu_id)
    nn.set_default_context(ctx)


def get_nnabla_context(gpu_id):
    global contexts
    if gpu_id in contexts:
        return contexts[gpu_id]
    if gpu_id < 0:
        ctx = get_extension_context('cpu')
    else:
        try:
            ctx = get_extension_context('cudnn', device_id=gpu_id)
        except ModuleNotFoundError:
            warnings.warn('Could not get CUDA context and cuDNN context. Fallback to CPU context instead',
                          RuntimeWarning)
            ctx = get_extension_context('cpu')
    contexts[gpu_id] = ctx
    return ctx
