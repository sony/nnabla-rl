# Copyright 2021 Sony Corporation.
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

import nnabla as nn
from nnabla_rl.algorithm import AlgorithmConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo


class SolverBuilder():
    """Solver builder interface class
    """

    def __call__(self,
                 env_info: EnvironmentInfo,
                 algorithm_config: AlgorithmConfig,
                 **kwargs) -> nn.solver.Solver:
        return self.build_solver(env_info, algorithm_config, **kwargs)

    def build_solver(self,
                     env_info: EnvironmentInfo,
                     algorithm_config: AlgorithmConfig,
                     **kwargs) -> nn.solver.Solver:
        """Build solver function

        Args:
            env_info (:py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):\
                environment information
            algorithm_config (:py:class:`AlgorithmConfig <nnabla_rl.algorithm.AlgorithmConfig>`): \
                configuration class of the target algorithm

        Returns:
            Solver: solver instance.
        """
        raise NotImplementedError
