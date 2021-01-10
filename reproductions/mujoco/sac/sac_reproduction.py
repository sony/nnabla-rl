import argparse

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed
from nnabla_rl.hook import as_hook
from nnabla_rl.utils import serializers


@as_hook(timing=100)
def print_iteration_number(algorithm):
    print('Current iteration: {}'.format(algorithm.iteration_num))


def select_total_iterations(env_name):
    if env_name in ['Hopper-v2']:
        total_iterations = 1000000
    elif env_name in ['Humanoid-v2']:
        total_iterations = 10000000
    else:
        total_iterations = 3000000
    print(f'Selected total iterations: {total_iterations}')
    return total_iterations


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=5000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=5000)

    train_env = build_mujoco_env(args.env, seed=args.seed, render=args.render)
    if args.snapshot_dir is None:
        params = A.SACParam(fix_temperature=args.fix_temperature)
        sac = A.SAC(train_env, params=params)
    else:
        sac = serializers.load_snapshot(args.snapshot_dir)
    hooks = [print_iteration_number, save_snapshot_hook, evaluation_hook]
    sac.set_hooks(hooks)

    total_iterations = select_total_iterations(args.env)
    sac.train_online(train_env, total_iterations=total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    sac = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(sac, A.SAC):
        raise ValueError('Loaded snapshot is not trained with SAC!')

    eval_env = build_mujoco_env(
        args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(sac, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)

    # SAC algorithm params
    parser.add_argument('--fix-temperature', action='store_true')

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
