import argparse

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env
from nnabla_rl.hook import as_hook
from nnabla_rl.utils import serializers


@as_hook(timing=100)
def print_iteration_number(algorithm):
    print('Current iteration: {}'.format(algorithm.iteration_num))


def select_start_timesteps(env_name):
    if env_name in ['Ant-v2', 'HalfCheetah-v2']:
        timesteps = 10000
    else:
        timesteps = 1000
    print(f'Selected start timesteps: {timesteps}')
    return timesteps


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=0)

    outdir = '{}_results'.format(args.env)

    eval_env = build_mujoco_env(
        args.env, test=True, seed=100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=5000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=5000)

    train_env = build_mujoco_env(args.env, seed=1, render=args.render)
    if args.snapshot_dir is None:
        timesteps = select_start_timesteps(args.env)
        params = A.TD3Param(start_timesteps=timesteps)
        td3 = A.TD3(train_env, params=params)
    else:
        td3 = serializers.load_snapshot(args.snapshot_dir)
    hooks = [print_iteration_number, save_snapshot_hook, evaluation_hook]
    td3.set_hooks(hooks)

    td3.train_online(train_env, total_iterations=1000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=0)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    td3 = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(td3, A.TD3):
        raise ValueError('Loaded snapshot is not trained with TD3!')

    eval_env = build_mujoco_env(
        args.env, test=True, seed=200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(td3, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
