# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import os

import numpy as np

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla_rl.logger import logger
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed
from nnabla_rl.writers import FileWriter


def run_training(args):
    set_global_seed(args.seed)
    train_env = build_atari_env(args.env, seed=args.seed)
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 100, render=args.render)

    iteration_num_hook = H.IterationNumHook(timing=100)

    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=250000, writer=writer)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=250000)

    actor_num = 16
    total_timesteps = 50000000
    config = A.A2CConfig(gpu_id=args.gpu, actor_num=actor_num)
    a2c = A.A2C(train_env, config=config)
    a2c.set_hooks(hooks=[iteration_num_hook, save_snapshot_hook, evaluation_hook])

    a2c.train(train_env, total_iterations=total_timesteps)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    config = A.A2CConfig(gpu_id=args.gpu)
    a2c = serializers.load_snapshot(args.snapshot_dir, config=config)
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
    parser.add_argument('--save-dir', type=str, default="")
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
