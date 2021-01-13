from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.algorithm import AlgorithmParam


class ReplayBufferBuilder():
    def __call__(self,
                 env_info: EnvironmentInfo,
                 algorithm_params: AlgorithmParam,
                 **kwargs) -> ReplayBuffer:
        return self.build_replay_buffer(env_info, algorithm_params, **kwargs)

    def build_replay_buffer(self,
                            env_info: EnvironmentInfo,
                            algorithm_params: AlgorithmParam,
                            **kwargs) -> ReplayBuffer:
        raise NotImplementedError
