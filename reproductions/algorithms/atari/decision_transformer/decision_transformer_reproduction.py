# Copyright 2023,2024 Sony Group Corporation.
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
import re

import numpy as np
from atari_dataset_loader import find_all_file_with_name, load_dataset_by_dataset_num

import nnabla.solvers as NS
import nnabla_rl as rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla.utils.learning_rate_scheduler import BaseLearningRateScheduler
from nnabla_rl.builders import LearningRateSchedulerBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.logger import logger
from nnabla_rl.replay_buffers import MemoryEfficientAtariTrajectoryBuffer
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed
from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm
from nnabla_rl.writers import FileWriter


class AtariDecaySolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        # Set initial alpha used internally in AdamW to 1.0
        solver = NS.AdamW(alpha=1.0, beta1=0.9, beta2=0.95, wd=algorithm_config.weight_decay)
        # Set true learning rate here
        solver.set_learning_rate(algorithm_config.learning_rate)
        return AutoClipGradByNorm(solver, algorithm_config.grad_clip_norm)


class AtariLearningRateScheduler(BaseLearningRateScheduler):
    def __init__(self, initial_learning_rate, context_length, batch_size, warmup_tokens, final_tokens):
        super().__init__()
        self._initial_learning_rate = initial_learning_rate
        self._context_length = context_length
        self._batch_size = batch_size
        self._warmup_tokens = warmup_tokens
        self._final_tokens = final_tokens
        self._processed_tokens = 0

    def get_learning_rate(self, iter):
        new_learning_rate = self._initial_learning_rate
        self._processed_tokens += self._context_length * self._batch_size

        if self._processed_tokens < self._warmup_tokens:
            new_learning_rate *= float(self._processed_tokens) / max(1, self._warmup_tokens)
        else:
            progress = float(self._processed_tokens - self._warmup_tokens) / max(
                1, self._final_tokens - self._warmup_tokens
            )
            new_learning_rate *= max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return new_learning_rate


class AtariLearningRateSchedulerBuilder(LearningRateSchedulerBuilder):
    def __init__(self, warmup_tokens, final_tokens) -> None:
        super().__init__()
        self._warmup_tokens = warmup_tokens
        self._final_tokens = final_tokens

    def build_scheduler(self, env_info, algorithm_config, **kwargs) -> BaseLearningRateScheduler:
        return AtariLearningRateScheduler(
            initial_learning_rate=algorithm_config.learning_rate,
            context_length=algorithm_config.context_length,
            batch_size=algorithm_config.batch_size,
            warmup_tokens=self._warmup_tokens,
            final_tokens=self._final_tokens,
        )


def num_datasets(dataset_path):
    return len(find_all_file_with_name(dataset_path, "observation"))


def get_next_trajectory(dataset, trajectory_length):
    s, a, r, t = dataset
    end = trajectory_length
    states = np.copy(s[:end])
    actions = np.copy(a[:end])
    rewards = np.copy(r[:end])
    non_terminals = np.copy(1 - t[:end])
    if end + 1 < len(s):
        next_states = np.copy(s[1 : end + 1])
    else:
        state_shape = s[0].shape
        next_states = np.concatenate((s[1:end], np.zeros(shape=(1, *state_shape), dtype=np.uint8)), axis=0)

    info = [{} for _ in range(trajectory_length)]
    for timestep, d in enumerate(info):
        d["rtg"] = np.sum(rewards[timestep:])
        d["timesteps"] = timestep
    assert all([len(data) == len(states) for data in (actions, rewards, non_terminals, next_states, info)])
    timesteps = len(info) - 1
    return list(zip(states, actions, rewards, non_terminals, next_states, info)), timesteps


def load_dataset(dataset_dir, buffer_size, context_length, trajectories_per_buffer):
    print(f"start loading dataset from: {dataset_dir}")
    # NOTE: actual number of loaded trajectories could be less than maximum possible trajectories
    max_possible_trajectories = buffer_size // context_length
    buffer = MemoryEfficientAtariTrajectoryBuffer(num_trajectories=max_possible_trajectories)

    max_timesteps = 1
    max_datasets = num_datasets(dataset_dir)
    dataset_seek_index = np.zeros(max_datasets, dtype=int)
    while len(buffer) < buffer_size:
        dataset_num = rl.random.drng.integers(low=0, high=max_datasets)
        print(f"loading dataset: #{dataset_num}")
        appended_trajectories = 0
        seek_index = dataset_seek_index[dataset_num]
        s, a, r, t = load_dataset_by_dataset_num(dataset_dir, dataset_num)
        while appended_trajectories < trajectories_per_buffer:
            s = s[seek_index:]
            a = a[seek_index:]
            r = r[seek_index:]
            t = t[seek_index:]
            if len(s) < context_length:
                print(f"all available trajectories in dataset #{dataset_num} has been loaded")
                break

            done_indices, *_ = np.where(t == 1)
            trajectory_length = done_indices[0] + 1

            if context_length <= trajectory_length:
                trajectory, timesteps = get_next_trajectory((s, a, r, t), trajectory_length)
                max_timesteps = max(max_timesteps, timesteps)
                buffer.append_trajectory(trajectory)
                appended_trajectories += 1
                print(f"loaded trajectories: {appended_trajectories}")

            # Set next index
            seek_index = trajectory_length
            dataset_seek_index[dataset_num] += trajectory_length
        print(f"loaded buffer size: {len(buffer)}")
    print(f"buffer size: {len(buffer)}, max timestep: {max_timesteps}")
    return buffer, max_timesteps


def get_target_return(env_name):
    if "Breakout" in env_name:
        return 90
    if "Seaquest" in env_name:
        return 1150
    if "Qbert" in env_name:
        return 14000
    if "Pong" in env_name:
        return 20
    raise NotImplementedError(f"No return is defined for: {env_name}")


def get_batch_size(env_name):
    if "Breakout" in env_name or "Seaquest" in env_name or "Qbert" in env_name:
        return 128
    if "Pong" in env_name:
        return 512
    raise NotImplementedError(f"No batch_size is defined for: {env_name}")


def get_context_length(env_name):
    if "Breakout" in env_name or "Seaquest" in env_name or "Qbert" in env_name:
        return 30
    if "Pong" in env_name:
        return 50
    raise NotImplementedError(f"No context_length is defined for: {env_name}")


def guess_dataset_path(env_name):
    game = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", env_name)[0]
    return f"datasets/{game}/1/replay_logs"


def run_training(args):
    outdir = f"{args.env}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    writer = FileWriter(outdir, "evaluation_result")
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 100, render=args.render, use_gymnasium=args.use_gymnasium
    )
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=args.eval_timing, writer=writer)

    epoch_num_hook = H.EpochNumHook(iteration_per_epoch=1)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    dataset_path = args.dataset_path if args.dataset_path is not None else guess_dataset_path(args.env)
    context_length = args.context_length if args.context_length is not None else get_context_length(args.env)
    dataset, max_timesteps = load_dataset(dataset_path, args.buffer_size, context_length, args.trajectories_per_buffer)

    final_tokens = 2 * len(dataset) * context_length * 3
    target_return = args.target_return if args.target_return is not None else get_target_return(args.env)
    batch_size = args.batch_size if args.batch_size is not None else get_batch_size(args.env)
    config = A.DecisionTransformerConfig(
        gpu_id=args.gpu,
        context_length=context_length,
        max_timesteps=max_timesteps,
        batch_size=batch_size,
        target_return=target_return,
    )
    env_info = EnvironmentInfo.from_env(eval_env)
    decision_transformer = A.DecisionTransformer(
        env_info,
        config=config,
        transformer_wd_solver_builder=AtariDecaySolverBuilder(),
        lr_scheduler_builder=AtariLearningRateSchedulerBuilder(args.warmup_tokens, final_tokens),
    )
    decision_transformer.set_hooks(hooks=[epoch_num_hook, save_snapshot_hook, evaluation_hook])

    print(f"total epochs: {args.total_epochs}")
    # decision transformer runs 1 epoch per iteration
    decision_transformer.train(dataset, total_iterations=args.total_epochs)

    eval_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_atari_env(
        args.env, test=True, seed=args.seed + 200, render=args.render, use_gymnasium=args.use_gymnasium
    )
    config = {"gpu_id": args.gpu}
    decision_transformer = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(decision_transformer, A.DecisionTransformer):
        raise ValueError("Loaded snapshot is not trained with DecisionTransformer!")

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    returns = evaluator(decision_transformer, eval_env)
    mean = np.mean(returns)
    std_dev = np.std(returns)
    median = np.median(returns)
    logger.info("Evaluation results. mean: {} +/- std: {}, median: {}".format(mean, std_dev, median))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--total-epochs", type=int, default=5)
    parser.add_argument("--trajectories-per-buffer", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--warmup-tokens", type=int, default=512 * 20)
    parser.add_argument("--save_timing", type=int, default=1)
    parser.add_argument("--eval_timing", type=int, default=1)
    parser.add_argument("--showcase_runs", type=int, default=10)
    parser.add_argument("--target-return", type=int, default=None)
    parser.add_argument("--use-gymnasium", action="store_true")

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
