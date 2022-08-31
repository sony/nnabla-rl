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

from dataclasses import dataclass
from typing import List

import numpy as np

from nnabla_rl.algorithms.ddp import DDP, DDPConfig
from nnabla_rl.numpy_models.cost_function import CostFunction
from nnabla_rl.numpy_models.dynamics import Dynamics


@dataclass
class iLQRConfig(DDPConfig):
    # same as DDP
    pass


class iLQR(DDP):
    '''iterative LQR (Linear Quadratic Regulator) algorithm.
    This class implements the iterative Linear Quadratic Requlator (iLQR) algorithm
    proposed by Y. Tassa, et al. in the paper:
    "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization"
    For details see: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        dynamics (:py:class:`Dynamics <nnabla_rl.non_nn_models.dynamics.Dynamics>`):
            dynamics of the system to control
        cost_function (:py:class:`Dynamics <nnabla_rl.non_nn_models.cost_function.CostFunction>`):
            cost function to optimize the trajectory
        config (:py:class:`iLQRConfig <nnabla_rl.algorithmss.ilqr.iLQRConfig>`):
            the parameter for iLQR controller
    '''
    _config: iLQRConfig

    def __init__(self,
                 env_or_env_info,
                 dynamics: Dynamics,
                 cost_function: CostFunction,
                 config=iLQRConfig()):
        super(iLQR, self).__init__(env_or_env_info, dynamics, cost_function, config=config)

    def _backward_pass(self, trajectory, dynamics, cost_function, mu):
        x_last, u_last = trajectory[-1]
        # Initialize Vx and Vxx to the gradient/hessian of value function of the final state of the trajectory
        Vx, *_ = cost_function.gradient(x_last, u_last, self._config.T_max, final_state=True)
        Vxx, *_ = cost_function.hessian(x_last, u_last, self._config.T_max, final_state=True)
        E = np.identity(n=Vxx.shape[0])

        ks: List[np.ndarray] = []
        Ks: List[np.ndarray] = []
        Qus: List[np.ndarray] = []
        Quus: List[np.ndarray] = []
        Quu_invs: List[np.ndarray] = []
        for t in reversed(range(self._config.T_max - 1)):
            (x, u) = trajectory[t]
            Cx, Cu = cost_function.gradient(x, u, self._config.T_max - t - 1)
            Cxx, Cxu, Cux, Cuu = cost_function.hessian(x, u, self._config.T_max - t - 1)

            Fx, Fu = dynamics.gradient(x, u, self._config.T_max - t - 1)
            # iLQR ignore dynamics' hessian

            Quu = Cuu + Fu.T.dot(Vxx + mu * E).dot(Fu)

            if not self._is_positive_definite(Quu):
                return ks, Ks, Qus, Quus, Quu_invs, False

            Qx = Cx + Fx.T.dot(Vx)
            Qu = Cu + Fu.T.dot(Vx)

            Qxx = Cxx + Fx.T.dot(Vxx).dot(Fx)
            # NOTE: Qxu and Qux should be symmetric and same matrix. i.e. Qxu = Qux and Qxu.T = Qux
            Qxu = Cxu + Fu.T.dot(Vxx + mu * E).dot(Fx).T
            Qux = Cux + Fu.T.dot(Vxx + mu * E).dot(Fx)
            assert np.allclose(Qxu, Qux.T)

            Quu_inv = np.linalg.inv(Quu)
            k = -Quu_inv.dot(Qu)
            K = -Quu_inv.dot(Qux)

            ks.append(k)
            Ks.append(K)
            Qus.append(Qu)
            Quus.append(Quu)
            Quu_invs.append(Quu_inv)

            Vx = Qx + K.T.dot(Quu).dot(k) + K.T.dot(Qu) + Qux.T.dot(k)
            Vxx = Qxx + K.T.dot(Quu).dot(K) + K.T.dot(Qux) + Qux.T.dot(K)

        ks = list(reversed(ks))
        Ks = list(reversed(Ks))
        Qus = list(reversed(Qus))
        Quus = list(reversed(Quus))
        Quu_invs = list(reversed(Quu_invs))

        return ks, Ks, Qus, Quus, Quu_invs, True
