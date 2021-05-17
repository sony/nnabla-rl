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

from multiprocessing import sharedctypes

import numpy as np


def mp_array_from_np_array(np_array):
    ctype = np.ctypeslib.as_ctypes_type(np_array.dtype)
    size = int(np.prod(np_array.shape))
    mp_array = sharedctypes.RawArray(ctype, size)
    return np_to_mp_array(np_array, mp_array, dtype=np_array.dtype)


def np_to_mp_array(np_array, mp_array, dtype):
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    mp_array = np.frombuffer(mp_array, dtype=ctype)
    mp_array[:] = np_array.flatten()
    return mp_array


def mp_to_np_array(mp_array, np_shape, dtype):
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    mp_array = np.frombuffer(mp_array, dtype=ctype)
    np_array = np.ctypeslib.as_array(mp_array).reshape(np_shape)
    return np_array


def new_mp_arrays_from_params(params):
    '''Converts nnabla's parameters to dictionary of multiprocessable arrays

    Args:
      params(OrderedDict): dictionary of parameters to convert to multiprocess arrays

    Returns: dict
      dictionary of multiprocess arrays with size same as corresponding parameter's size
    '''
    arrays = {}
    for key in params.keys():
        param_np_array = params[key].d
        # FIXME: cast to float32.
        # This is a workaround for compensating nnabla's parameter initialization with float64
        param_np_array = param_np_array.astype(np.float32)
        arrays[key] = mp_array_from_np_array(param_np_array)
    return arrays


def copy_params_to_mp_arrays(params, mp_arrays):
    '''Copy nnabla's parameters to multiprocessable arrays

    Args:
      params(OrderedDict): dictionary of parameters to convert to multiprocess arrays
      mp_arrays(dict): dictionary of multiprocess arrays with keys same as corresponding parameter's key
    '''
    for key in params.keys():
        np_array = params[key].d
        # FIXME: cast to float32.
        # This is a workaround for compensating nnabla's parameter initialization with float64
        np_array = np_array.astype(np.float32)
        mp_array = mp_arrays[key]
        mp_array = np_to_mp_array(np_array, mp_array, dtype=np_array.dtype)


def copy_mp_arrays_to_params(mp_arrays, params):
    '''Copy nnabla's parameters from multiprocessable arrays

    Args:
      mp_arrays(dict): dictionary of multiprocess arrays with keys same as corresponding parameter's key
      params(OrderedDict): dictionary of parameters to convert to multiprocess arrays
    '''
    for key, mp_array in mp_arrays.items():
        param_shape = params[key].shape
        # FIXME: force using float32.
        # This is a workaround for compensating nnabla's parameter initialization with float64
        np_array = mp_to_np_array(
            mp_array, np_shape=param_shape, dtype=np.float32)
        params[key].d = np_array
