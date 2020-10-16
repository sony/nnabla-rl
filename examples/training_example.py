import gym

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.hook import as_hook


@as_hook(timing=100)
def print_iteration_number(algorithm):
    print('Current iteration: {}'.format(algorithm.iteration_num))


def build_env(seed=None):
    env = gym.make('Pendulum-v0')
    env = NumpyFloat32Env(env)
    env = ScreenRenderEnv(env)
    env.seed(seed)
    return env


def main():
    # Run on gpu if possible.
    nnabla_rl.run_on_gpu(0)

    # Evaluate the trained network (Optional)
    eval_env = build_env(seed=100)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        timing=1000,
        writer=W.FileWriter(outdir='./pendulum_v0_ddpg_results',
                            file_prefix='evaluation_result'))

    # Save trained algorithm snapshot (Optional)
    save_snapshot_hook = H.SaveSnapshotHook(
        './pendulum_v0_ddpg_results', timing=1000)

    train_env = build_env()
    params = A.DDPGParam(start_timesteps=200)
    ddpg = A.DDPG(train_env, params=params)
    ddpg.set_hooks(hooks=[print_iteration_number,
                          save_snapshot_hook,
                          evaluation_hook])

    ddpg.train(train_env, total_iterations=10000)

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
