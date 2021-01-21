from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.algorithm import AlgorithmParam


class PreprocessorBuilder():
    def __call__(self,
                 scope_name: str,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> Preprocessor:
        return self.build_preprocessor(scope_name, env_info, algorithm_params, **kwargs)

    def build_preprocessor(self,
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_params: AlgorithmParam,
                           **kwargs) -> Preprocessor:
        raise NotImplementedError
