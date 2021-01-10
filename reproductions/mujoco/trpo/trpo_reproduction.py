import argparse

import numpy as np

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed
from nnabla_rl.logger import logger
from nnabla_rl.utils import serializers


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=10000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=10000)
    iteration_num_hook = H.IterationNumHook(timing=5000)

    train_env = build_mujoco_env(args.env, seed=args.seed, render=args.render)
    if args.snapshot_dir is None:
        trpo = A.TRPO(train_env)
    else:
        trpo = serializers.load_snapshot(args.snapshot_dir)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    trpo.set_hooks(hooks)

    trpo.train_online(train_env, total_iterations=1000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpou)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    trpo = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(trpo, A.TRPO):
        raise ValueError('Loaded snapshot is not trained with TRPO!')

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 200, render=False)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    returns = evaluator(trpo, eval_env)
    mean = np.mean(returns)
    std_dev = np.std(returns)
    median = np.median(returns)
    logger.info('Evaluation results. mean: {} +/- std: {}, median: {}'.format(mean, std_dev, median))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
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
