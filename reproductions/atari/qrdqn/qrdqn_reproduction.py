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

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.replay_buffers as RB
import nnabla_rl.writers as W
from nnabla_rl.builders import ReplayBufferBuilder
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed


class MemoryEfficientBufferBuilder(ReplayBufferBuilder):
    def __call__(self, env_info, algorithm_config, **kwargs):
        return RB.MemoryEfficientAtariBuffer(capacity=algorithm_config.replay_buffer_size)


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_atari_env(args.env, test=True, seed=args.seed + 100, render=args.render)
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=250000, writer=W.FileWriter(outdir=outdir, file_prefix='evaluation_result'))

    iteration_num_hook = H.IterationNumHook(timing=100)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=50000)

    train_env = build_atari_env(args.env, seed=args.seed, render=args.render)
    config = A.QRDQNConfig(gpu_id=args.gpu)
    qrdqn = A.QRDQN(train_env, config=config, replay_buffer_builder=MemoryEfficientBufferBuilder())

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    qrdqn.set_hooks(hooks)

    qrdqn.train_online(train_env, total_iterations=50000000)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    config = A.QRDQNConfig(gpu_id=args.gpu)
    qrdqn = serializers.load_snapshot(args.snapshot_dir, config=config)
    if not isinstance(qrdqn, A.QRDQN):
        raise ValueError('Loaded snapshot is not trained with QRDQN!')

    eval_env = build_atari_env(args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(qrdqn, eval_env)


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
