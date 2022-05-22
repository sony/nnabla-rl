# Copyright 2022 Sony Group Corporation.
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

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from nnabla_rl.numpy_models.numpy_model import NumpyModel


class Dynamics(NumpyModel, metaclass=ABCMeta):
    '''Base dynamics
    '''

    def __init__(self):
        pass

    def state_dim(self) -> int:
        raise NotImplementedError

    def action_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def next_state(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        '''predict next state. if the dynamics is probabilistic, will return the mean of the next_state

        .. math::
            x_{t+1} = D(x_{t}, u_{t})

        Args:
            x (np.ndarray): State
            u (np.ndarray): Action
            t (int): timestep
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]] : Predicted next (mean) state and info
                Returned arrays has (batch_size, state_shape) if batched is True.
        '''
        raise NotImplementedError

    def gradient(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        ''' gradient of the dynamics with respect to the state and action

        .. math::
            D_{x} &= {\nabla}_{x}D(x_{t}, u_{t})
            D_{u} &= {\nabla}_{u}D(x_{t}, u_{t})

        Args:
            x (np.ndarray): state.
            u (np.ndarray): action.
            t (int): timestep
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                gradient of the dynamics with respect to given state and action.
                Tuple is in the order of Dx, Du.
                Returned arrays has (batch_size, gradient_shape) if batched is True.
                (NOTE: This will NOT compute the gradeint of probability density)
        '''
        raise NotImplementedError

    def hessian(self, x: np.ndarray, u: np.ndarray, t: int, batched: bool = False) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''' hessian of the dynamics with respect to the state and action

        .. math::
            D_{xx} &= {\nabla}^{2}_{x}D(x_{t}, u_{t})
            D_{xu} &= {\nabla}_{u}\nabla}_{x}D(x_{t}, u_{t})
            D_{ux} &= {\nabla}_{x}\nabla}_{u}D(x_{t}, u_{t})
            D_{uu} &= {\nabla}^{2}_{u}D(x_{t}, u_{t})

        Args:
            x (np.ndarray): state.
            u (np.ndarray): action.
            t (int): timestep
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                hessian of the dynamics with respect to given state and action.
                Tuple is in the order of Dxx, Dxu, Dux, Duu.
                Returned arrays has (batch_size, hessian_shape) if batched is True.
                (NOTE: This will NOT compute the hessian of probability density)
        '''
        raise NotImplementedError
