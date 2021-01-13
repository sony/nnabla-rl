import nnabla as nn

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.algorithm import AlgorithmParam


class SolverBuilder():
    def __call__(self,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> nn.solver.Solver:
        return self.build_solver(env_info, algorithm_params, **kwargs)

    def build_solver(self,
                     env_info: EnvironmentInfo,
                     algorithm_params: AlgorithmParam,
                     **kwargs) -> nn.solver.Solver:
        raise NotImplementedError
