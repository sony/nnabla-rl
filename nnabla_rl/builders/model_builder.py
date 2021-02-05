# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from nnabla_rl.models import Model, VFunction, QFunction, RewardFunction, StochasticPolicy, DeterministicPolicy, \
    VariationalAutoEncoder, StateActionQuantileFunction, QuantileDistributionFunction, ValueDistributionFunction, \
    Perturbator
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.algorithm import AlgorithmParam


class ModelBuilder():
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> Model:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> Model:
        raise NotImplementedError


class VFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> VFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> VFunction:
        raise NotImplementedError


class QFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> QFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> QFunction:
        raise NotImplementedError


class RewardFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> RewardFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> RewardFunction:
        raise NotImplementedError


class StochasticPolicyBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> StochasticPolicy:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> StochasticPolicy:
        raise NotImplementedError


class DeterministicPolicyBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> DeterministicPolicy:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> DeterministicPolicy:
        raise NotImplementedError


class StateActionQuantileFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> StateActionQuantileFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> StateActionQuantileFunction:
        raise NotImplementedError


class ValueDistributionFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> ValueDistributionFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> ValueDistributionFunction:
        raise NotImplementedError


class QuantileDistributionFunctionBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> QuantileDistributionFunction:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> QuantileDistributionFunction:
        raise NotImplementedError


class VariationalAutoEncoderBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> VariationalAutoEncoder:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> VariationalAutoEncoder:
        raise NotImplementedError


class PerturbatorBuilder(ModelBuilder):
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> Perturbator:
        return self.build_model(scope_name, env_info, algorithm_params, **kwargs)

    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> Perturbator:
        raise NotImplementedError
