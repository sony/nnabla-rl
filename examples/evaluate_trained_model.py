import gym

from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator

from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv


def build_env():
    env = gym.make('Pendulum-v0')
    env = NumpyFloat32Env(env)
    env = ScreenRenderEnv(env)
    return env


def main():
    snapshot_dir = './pendulum_v0_snapshot/iteration-10000'
    algorithm = serializers.load_snapshot(snapshot_dir)

    env = build_env()

    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluator(algorithm, env)

    env.close()


if __name__ == "__main__":
    main()
