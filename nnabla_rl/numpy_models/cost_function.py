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
from typing import Iterable, Optional, Tuple, cast

import numpy as np

from nnabla_rl.numpy_models.numpy_model import NumpyModel


class CostFunction(NumpyModel, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(
            self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        ''' evaluate cost for given state and action

        Args:
            x (np.ndarray): state
            u (Optional[np.ndarray]): action
            t (int): timestep
            final_state (bool): flag whether final timestep state or not. Defaults to False.
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            np.ndarray: cost. Returned arrays has (batch_size, 1) if batched is True.
        '''
        raise NotImplementedError

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ''' gradient of cost with respect to the states and actions in sequence form

        Args:
            x (np.ndarray): state.
            u (Optional[np.ndarray]): action.
            t (int): timestep
            final_state (bool): flag whether final timestep state or not. Defaults to False.
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                gradient of the cost with respect to given state and action.
                Tuple is in the order of Cx, Cu.
                Returned arrays has (batch_size, gradient_shape) if batched is True.
        '''
        raise NotImplementedError

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        ''' hessian of the cost with respect to given state and action

        Args:
            x (np.ndarray): state.
            u (Optional[np.ndarray]): action.
            t (int): timestep
            final_state (bool): flag whether final timestep state or not. Defaults to False.
            batched (bool): Turn this flag to true if input state and action is batched.
                When True, x and u should have size (batch_size, state_shape) and (batch_size, action_shape)
                respectively.
                Default is False.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
                hessian of the cost with respect to given state and action.
                Tuple is in the order of Cxx, Cxu, Cux, Cuu.
                Returned arrays has (batch_size, hessian_shape) if batched is True.
        '''
        raise NotImplementedError

    def __add__(self, o):
        if not isinstance(o, CostFunction):
            raise ValueError('Only cost function can be added together')
        return SumCost([self, o])


class SumCost(CostFunction):
    def __init__(self, cost_functions: Iterable[CostFunction]) -> None:
        self._cost_functions = cost_functions

    def evaluate(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> np.ndarray:
        cost = np.zeros(1)
        for cost_function in self._cost_functions:
            cost += cost_function.evaluate(x, u, t, final_state=final_state, batched=batched)
        return cost

    def gradient(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Cx = np.zeros_like(x)
        if final_state:
            for cost_function in self._cost_functions:
                cx, cu = cost_function.gradient(x, u, t, final_state=final_state, batched=batched)
                Cx += cx
            Cu = None
        else:
            assert isinstance(u, np.ndarray)
            Cu = np.zeros_like(u)
            for cost_function in self._cost_functions:
                cx, cu = cost_function.gradient(x, u, t, final_state=final_state, batched=batched)
                Cx += cx
                Cu += cu
        return Cx, Cu

    def hessian(
        self, x: np.ndarray, u: Optional[np.ndarray], t: int, final_state: bool = False, batched: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        batch_size = x.shape[0] if batched else -1
        x_dim = x.shape[1] if batched else x.shape[0]
        Cxx = np.zeros((batch_size, x_dim, x_dim)) if batched else np.zeros((x_dim, x_dim))
        if final_state:
            Cxu = None
            Cux = None
            Cuu = None
            for cost_function in self._cost_functions:
                cxx, cxu, cux, cuu = cost_function.hessian(x, u, t, final_state=final_state, batched=batched)
                Cxx += cxx
        else:
            assert isinstance(u, np.ndarray)
            u_dim = u.shape[0]
            Cxu = np.zeros((batch_size, x_dim, u_dim)) if batched else np.zeros((x_dim, u_dim))
            Cux = np.zeros((batch_size, u_dim, x_dim)) if batched else np.zeros((u_dim, x_dim))
            Cuu = np.zeros((batch_size, u_dim, u_dim)) if batched else np.zeros((u_dim, u_dim))
            for cost_function in self._cost_functions:
                cxx, cxu, cux, cuu = cost_function.hessian(x, u, t, final_state=final_state, batched=batched)
                Cxx += cxx
                Cxu += cast(np.ndarray, cxu)
                Cux += cast(np.ndarray, cux)
                Cuu += cast(np.ndarray, cuu)
        return Cxx, Cxu, Cux, Cuu
