import argparse

import gym

import numpy as np

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.utils.context as context
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.hook import as_hook
from nnabla_rl.logger import logger
from nnabla_rl.utils import serializers
from nnabla_rl.utils.reproductions import build_atari_env
from nnabla_rl.writers import FileWriter


@as_hook(timing=1)
def print_iteration_number(algorithm):
    print('Current iteration: {}'.format(algorithm.iteration_num))


def run_training(args):
    nnabla_rl.run_on_gpu()

    eval_env = build_atari_env(
        args.env, test=True, seed=100, render=args.render)

    outdir = '{}_results'.format(args.env)
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicEvaluator()
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=2, writer=writer)

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=100)

    train_env = build_atari_env(args.env, seed=1, render=args.render)
    if args.snapshot_dir is None:
        trpo = A.ICML2015TRPO(train_env)
    else:
        trpo = serializers.load_snapshot(args.snapshot_dir)
    hooks = [print_iteration_number, save_snapshot_hook, evaluation_hook]
    trpo.set_hooks(hooks)

    trpo.train_online(train_env, total_iterations=500)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=0)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    trpo = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(trpo, A.ICML2015TRPO):
        raise ValueError('Loaded snapshot is not trained with ICML2015TRPO')

    eval_env = build_atari_env(
        args.env, test=True, seed=200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(trpo, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='BeamRiderNoFrameskip-v4')
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
