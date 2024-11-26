# Copyright 2024 Sony Group Corporation.
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
from option_critic_utils import build_option_critic_atari_env

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.replay_buffers as RB
import nnabla_rl.writers as W
from nnabla_rl.builders import ReplayBufferBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.logger import logger
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator
from nnabla_rl.utils.reproductions import set_global_seed
from nnabla_rl.writers import FileWriter


class MemoryEfficientBufferBuilder(ReplayBufferBuilder):
    def __call__(self, env_info: EnvironmentInfo, algorithm_config: A.OptionCriticConfig, **kwargs):
        return RB.MemoryEfficientAtariBuffer(capacity=algorithm_config.replay_buffer_size)


def advantage_offset_value(env_name):
    # We have found that the offset value impacts the score,
    # and we are currently investigating this issue.
    if "Zaxxon" in env_name:
        return 0.0125
    elif "MsPacman" in env_name:
        return 0.075
    else:
        return 0.01


def run_training(args):
    outdir = f"{args.env}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    writer = FileWriter(outdir, "evaluation_result")
    eval_env = build_option_critic_atari_env(
        args.env,
        test=True,
        seed=args.seed + 100,
        render=args.render,
        use_gymnasium=args.use_gymnasium,
    )
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=args.eval_timing, writer=writer)

    iteration_num_hook = H.IterationNumHook(timing=10000)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_state_hook = H.IterationStateHook(
        writer=W.FileWriter(outdir=outdir, file_prefix="iteration_state", fmt="%.5f"),
        timing=args.iteration_state_timing,
        start_timing=50000,
    )

    train_env = build_option_critic_atari_env(args.env, seed=args.seed, use_gymnasium=args.use_gymnasium)

    config = A.OptionCriticConfig(gpu_id=args.gpu, advantage_offset=advantage_offset_value(args.env))
    option_critic = A.OptionCritic(train_env, config=config, replay_buffer_builder=MemoryEfficientBufferBuilder())
    option_critic.set_hooks(hooks=[iteration_num_hook, save_snapshot_hook, evaluation_hook, iteration_state_hook])

    option_critic.train(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_option_critic_atari_env(
        args.env, test=True, seed=args.seed + 200, render=args.render, use_gymnasium=args.use_gymnasium
    )
    config = A.OptionCriticConfig(gpu_id=args.gpu)
    option_critic = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(option_critic, A.OptionCritic):
        raise ValueError("Loaded snapshot is not trained with OptionCritic!")

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    returns = evaluator(option_critic, eval_env)
    mean = np.mean(returns)
    std_dev = np.std(returns)
    median = np.median(returns)
    logger.info("Evaluation results. mean: {} +/- std: {}, median: {}".format(mean, std_dev, median))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="AsterixNoFrameskip-v4")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--total_iterations", type=int, default=50000000)
    parser.add_argument("--save_timing", type=int, default=250000)
    parser.add_argument("--eval_timing", type=int, default=250000)
    parser.add_argument("--showcase_runs", type=int, default=10)
    parser.add_argument("--iteration_state_timing", type=int, default=10000)
    parser.add_argument("--use-gymnasium", action="store_true")

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
