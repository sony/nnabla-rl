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
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed
from nnabla_rl.writers import FileWriter


def run_training(args):
    outdir = f"{args.env}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    writer = FileWriter(outdir, "evaluation_result")
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 100, render=args.render, use_gymnasium=args.use_gymnasium
    )
    evaluator = EpisodicEvaluator()
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=args.eval_timing, writer=writer)

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_num_hook = H.IterationNumHook(timing=int(1e5))

    train_env = build_atari_env(args.env, seed=args.seed, render=args.render, use_gymnasium=args.use_gymnasium)

    config = A.ICML2015TRPOConfig(gpu_id=args.gpu, gpu_batch_size=args.gpu_batch_size)
    trpo = A.ICML2015TRPO(train_env, config=config)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    trpo.set_hooks(hooks)

    trpo.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 200, render=args.render, use_gymnasium=args.use_gymnasium
    )
    config = A.ICML2015TRPOConfig(gpu_id=args.gpu)
    trpo = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(trpo, A.ICML2015TRPO):
        raise ValueError("Loaded snapshot is not trained with ICML2015TRPO")

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(trpo, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_batch_size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--total_iterations", type=int, default=50000000)
    parser.add_argument("--save_timing", type=int, default=100000)
    parser.add_argument("--eval_timing", type=int, default=100000)
    parser.add_argument("--showcase_runs", type=int, default=10)
    parser.add_argument("--use-gymnasium", action="store_true")

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
