import argparse

import gym

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed, d4rl_dataset_to_buffer
from nnabla_rl.utils import serializers


def select_mmd_sigma(env_name, mmd_kernel):
    if mmd_kernel == 'gaussian':
        mmd_sigma = 20.0
    elif mmd_kernel == 'laplacian':
        mmd_sigma = 20.0 if 'walker2d' in env_name else 10.0
    else:
        raise ValueError(f'Unknown mmd kernel: {mmd_kernel}')
    print(f'selected mmd sigma: {mmd_sigma}')
    return mmd_sigma


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_{args.mmd_kernel}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=5000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=5000)
    iteration_num_hook = H.IterationNumHook(timing=100)
    iteration_state_hook = H.IterationStateHook(timing=100)

    train_env = gym.make(args.env)
    train_dataset = train_env.get_dataset()
    buffer = d4rl_dataset_to_buffer(train_dataset, max_buffer_size=1000000)
    if args.snapshot_dir is None:
        mmd_sigma = select_mmd_sigma(args.env, args.mmd_kernel)
        params = A.BEARParam(mmd_sigma=mmd_sigma, mmd_type=args.mmd_kernel)
        bear = A.BEAR(train_env, params=params)
    else:
        bear = serializers.load_snapshot(args.snapshot_dir)
    hooks = [save_snapshot_hook, evaluation_hook,
             iteration_num_hook, iteration_state_hook]
    bear.set_hooks(hooks)

    bear.train_offline(buffer, total_iterations=1000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    bear = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(bear, A.BEAR):
        raise ValueError('Loaded snapshot is not trained with BEAR!')

    eval_env = build_mujoco_env(
        args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(bear, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ant-expert-v0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--mmd-kernel', type=str,
                        default="gaussian", choices=["laplacian", "gaussian"])

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
