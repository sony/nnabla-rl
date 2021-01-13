from nnabla_rl.models import Model
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
