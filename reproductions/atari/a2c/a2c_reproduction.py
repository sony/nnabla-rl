import numpy as np
import argparse

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla_rl.utils.evaluator import TimestepEvaluator, EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_atari_env
from nnabla_rl.utils import serializers
from nnabla_rl.writers import FileWriter
from nnabla_rl.logger import logger
from nnabla_rl.utils.reproductions import set_global_seed


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    set_global_seed(args.seed)
    train_env = build_atari_env(args.env, seed=args.seed)
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 100, render=args.render)

    iteration_num_hook = H.IterationNumHook(timing=100)

    outdir = f'{args.env}_results/seed-{args.seed}'
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=250000, writer=writer)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=250000)

    actor_num = 8
    total_timesteps = 50000000
    params = A.A2CParam(actor_num=actor_num)
    a2c = A.A2C(train_env, params=params)
    a2c.set_hooks(hooks=[iteration_num_hook, save_snapshot_hook, evaluation_hook])

    a2c.train(train_env, total_iterations=total_timesteps)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    a2c = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(a2c, A.A2C):
        raise ValueError('Loaded snapshot is not trained with A2C!')

    eval_env = build_atari_env(args.env, test=True, seed=args.seed + 200, render=False)
    evaluator = EpisodicEvaluator(run_per_evaluation=30)
    returns = evaluator(a2c, eval_env)
    mean = np.mean(returns)
    std_dev = np.std(returns)
    median = np.median(returns)
    logger.info('Evaluation results. mean: {} +/- std: {}, median: {}'.format(mean, std_dev, median))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
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
