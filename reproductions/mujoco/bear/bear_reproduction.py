# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import gym

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, d4rl_dataset_to_experiences, set_global_seed


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

    buffer = ReplayBuffer(capacity=1000000)
    experiences = d4rl_dataset_to_experiences(train_dataset, size=buffer.capacity)
    buffer.append_all(experiences)

    mmd_sigma = select_mmd_sigma(args.env, args.mmd_kernel)
    config = A.BEARConfig(gpu_id=args.gpu, mmd_sigma=mmd_sigma, mmd_type=args.mmd_kernel)
    bear = A.BEAR(train_env, config=config)

    hooks = [save_snapshot_hook, evaluation_hook,
             iteration_num_hook, iteration_state_hook]
    bear.set_hooks(hooks)

    bear.train_offline(buffer, total_iterations=1000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    config = A.BEARConfig(gpu_id=args.gpu)
    bear = serializers.load_snapshot(args.snapshot_dir, config=config)
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
