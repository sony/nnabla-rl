import argparse

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.replay_buffers as RB
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import TimestepEvaluator, EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_atari_env
from nnabla_rl.hook import as_hook
from nnabla_rl.utils import serializers


@as_hook(timing=100)
def print_iteration_number(algorithm):
    print('Current iteration: {}'.format(algorithm.iteration_num))


def memory_efficient_buffer_builder(capacity):
    return RB.MemoryEfficientAtariBuffer(capacity=capacity)


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_results/seed-{args.seed}'

    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 100, render=args.render)
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=250000, writer=W.FileWriter(outdir=outdir,
                                                                file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=50000)

    train_env = build_atari_env(args.env, seed=args.seed, render=args.render)
    if args.snapshot_dir is None:
        iqn = A.IQN(train_env,
                    replay_buffer_builder=memory_efficient_buffer_builder)
    else:
        iqn = serializers.load_snapshot(args.snapshot_dir)
    hooks = [print_iteration_number, save_snapshot_hook, evaluation_hook]
    iqn.set_hooks(hooks)

    iqn.train_online(train_env, total_iterations=50000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    iqn = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(iqn, A.IQN):
        raise ValueError('Loaded snapshot is not trained with IQN!')
    iqn.update_algorithm_params(**{'test_epsilon': 0.001})

    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(iqn, eval_env)


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
