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

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed, d4rl_dataset_to_experiences
from nnabla_rl.utils import serializers
from nnabla_rl.replay_buffers import ReplacementSamplingReplayBuffer


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_datasetsize-{args.datasetsize}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=50000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=50000)
    iteration_num_hook = H.IterationNumHook(timing=50000)

    train_env = build_mujoco_env(args.env, seed=args.seed, render=args.render)
    train_dataset = train_env.get_dataset()

    expert_buffer = ReplacementSamplingReplayBuffer(capacity=args.datasetsize)
    expert_experiences = d4rl_dataset_to_experiences(train_dataset, size=args.datasetsize)
    expert_buffer.append_all(expert_experiences)

    if args.snapshot_dir is None:
        gail = A.GAIL(train_env, expert_buffer)
    else:
        gail = serializers.load_snapshot(args.snapshot_dir)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    gail.set_hooks(hooks)

    gail.train_online(train_env, total_iterations=25000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    gail = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(gail, A.GAIL):
        raise ValueError('Loaded snapshot is not trained with GAIL!')

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(gail, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-expert-v1')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasetsize', type=int, default=4000)
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
