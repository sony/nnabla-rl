# Copyright 2022,2023,2024 Sony Group Corporation.
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
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed

import sparse_mujoco  # noqa


def select_start_timesteps(env_name):
    if env_name in ["SparseAnt-v1", "SparseHalfCheetah-v1"]:
        timesteps = 10000
    else:
        timesteps = 1000
    print(f"Selected start timesteps: {timesteps}")
    return timesteps


def select_total_iterations(env_name):
    if env_name in ["SparseAnt-v1"]:
        total_iterations = 3000000
    else:
        total_iterations = 1000000
    print(f"Selected total iterations: {total_iterations}")
    return total_iterations


def select_alpha_pi(env_name):
    if env_name in ["SparseHopper-v1"]:
        alpha_pi = 0.04
    elif env_name in ["SparseHalfCheetah-v1"]:
        alpha_pi = 0.02
    elif env_name in ["SparseWalker2d-v1"]:
        alpha_pi = 0.02
    elif env_name in ["SparseAnt-v1"]:
        alpha_pi = 0.01
    else:
        alpha_pi = 1.0
    print(f"Selected alpha_pi: {alpha_pi}")
    return alpha_pi


def select_alpha_q(env_name):
    if env_name in ["SparseHopper-v1"]:
        alpha_q = 1.0
    elif env_name in ["SparseHalfCheetah-v1"]:
        alpha_q = 2.0
    elif env_name in ["SparseWalker2d-v1"]:
        alpha_q = 0.5
    elif env_name in ["SparseAnt-v1"]:
        alpha_q = 0.2
    else:
        alpha_q = 1.0
    print(f"Selected alpha_q: {alpha_q}")
    return alpha_q


def run_training(args):
    outdir = f"{args.env}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        evaluator,
        timing=args.eval_timing,
        writer=W.FileWriter(outdir=outdir, file_prefix="evaluation_result"),
    )

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_num_hook = H.IterationNumHook(timing=100)

    train_env = build_mujoco_env(args.env, seed=args.seed, render=args.render)
    timesteps = select_start_timesteps(args.env)
    alpha_pi = select_alpha_pi(args.env)
    alpha_q = select_alpha_q(args.env)
    config = A.MMESACConfig(gpu_id=args.gpu, start_timesteps=timesteps, alpha_pi=alpha_pi, alpha_q=alpha_q)
    mme_sac = A.MMESAC(train_env, config=config)

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    mme_sac.set_hooks(hooks)

    total_iterations = select_total_iterations(args.env) if args.total_iterations is None else args.total_iterations
    mme_sac.train_online(train_env, total_iterations=total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 200, render=args.render)
    config = A.MMESACConfig(gpu_id=args.gpu)
    mme_sac = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(mme_sac, A.MMESAC):
        raise ValueError("Loaded snapshot is not trained with MMESAC!")

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(mme_sac, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SparseAnt-v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--save_timing", type=int, default=5000)
    parser.add_argument("--eval_timing", type=int, default=5000)
    parser.add_argument("--showcase_runs", type=int, default=10)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
