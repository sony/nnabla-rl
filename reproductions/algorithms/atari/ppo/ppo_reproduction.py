# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    eval_env = build_atari_env(args.env, test=True, seed=args.seed + 100, render=args.render,
                               use_gymnasium=args.use_gymnasium)
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=args.evaluate_timing, writer=W.FileWriter(outdir=outdir,
                                                                              file_prefix='evaluation_result'))
    iteration_num_hook = H.IterationNumHook(timing=100)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    actor_num = 8

    train_env = build_atari_env(args.env, seed=args.seed, render=args.render, use_gymnasium=args.use_gymnasium)
    config = A.PPOConfig(gpu_id=args.gpu,
                         actor_num=actor_num,
                         total_timesteps=args.total_iterations,
                         timelimit_as_terminal=True,
                         seed=args.seed,
                         preprocess_state=False)
    ppo = A.PPO(train_env, config=config)

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    ppo.set_hooks(hooks)

    ppo.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    eval_env = build_atari_env(args.env, test=True, seed=args.seed + 200, render=args.render,
                               use_gymnasium=args.use_gymnasium)
    config = A.PPOConfig(gpu_id=args.gpu,
                         timelimit_as_terminal=True,
                         seed=args.seed,
                         preprocess_state=False)
    ppo = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(ppo, A.PPO):
        raise ValueError('Loaded snapshot is not trained with PPO!')

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(ppo, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--save-dir', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=10000000)
    parser.add_argument('--save_timing', type=int, default=250000)
    parser.add_argument('--evaluate_timing', type=int, default=250000)
    parser.add_argument('--showcase_runs', type=int, default=10)
    parser.add_argument('--use-gymnasium', action='store_true')

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
