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
from tqdm import tqdm

import nnabla.solvers as NS
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla.utils.learning_rate_scheduler import CosineScheduler
from nnabla_rl.builders import SolverBuilder
from nnabla_rl.environments.dummy import DummyMujocoEnv
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, d4rl_dataset_to_experiences, set_global_seed
from nnabla_rl.utils.solver_wrappers import AutoLearningRateScheduler

try:
    # import at the end. d4rl overrides logger unexpectedly
    import d4rl  # noqa
except ModuleNotFoundError:
    # Ignore if d4rl is not installed
    pass


class CosineDecayPolicySolverBuilder(SolverBuilder):
    def __init__(self, max_iterations):
        super().__init__()
        self._max_iterations = max_iterations

    def build_solver(self, env_info, algorithm_config, **kwargs):
        scheduler = CosineScheduler(algorithm_config.learning_rate, self._max_iterations)
        solver = NS.Adam(alpha=algorithm_config.learning_rate)
        solver = AutoLearningRateScheduler(solver, scheduler)
        return solver


def build_env_and_dataset(env_name, seed=None):
    d4rl_env = build_mujoco_env(env_name, seed=seed)
    if env_name == "FakeMujocoNNablaRL-v1":
        # Dummy environment
        d4rl_dataset = d4rl_env.get_qlearning_dataset()
    else:
        d4rl_dataset = d4rl.qlearning_dataset(d4rl_env)
    d4rl_dataset = clip_actions_in_dataset(d4rl_dataset)
    d4rl_dataset = normalize_dataset_score(d4rl_dataset)
    return d4rl_env, d4rl_dataset


def clip_actions_in_dataset(dataset, eps: float = 1e-5):
    lim = 1.0 - eps
    dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
    return dataset


def normalize_dataset_score(dataset):
    def dataset_to_trajectories(dataset):
        states = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        next_states = dataset["next_observations"]

        def same_observation(obs1, obs2):
            return np.linalg.norm(obs1 - obs2) <= 1e-6

        trajectories = []
        trajectory = []
        for i in tqdm(range(len(states))):
            trajectory.append((states[i], actions[i], rewards[i], terminals[i], next_states[i]))
            if i == len(states) - 1:
                continue
            if terminals[i] == 1.0 or (not same_observation(states[i + 1], next_states[i])):
                trajectories.append(trajectory)
                trajectory = []
        if len(trajectory) != 0:
            trajectories.append(trajectory)
        assert len(trajectories) > 1
        return trajectories

    def max_min_returns(trajectories):
        episode_returns = []
        for trajectory in trajectories:
            total_return = 0
            for _, _, r, _, _ in trajectory:
                total_return += r
            episode_returns.append(total_return)
        episode_returns.sort()
        return episode_returns[-1], episode_returns[0]

    trajectories = dataset_to_trajectories(dataset)
    max_return, min_return = max_min_returns(trajectories)
    print(f"len trajectories: {len(trajectories)}")
    print(f"max return: {max_return}, min return: {min_return}")

    dataset["rewards"] /= max_return - min_return
    dataset["rewards"] *= 1000.0

    return dataset


def run_training(args):
    outdir = f"{args.env}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)

    eval_env, dataset = build_env_and_dataset(args.env, seed=args.seed)
    dataset_size = dataset["observations"].shape[0]
    experiences = d4rl_dataset_to_experiences(dataset, size=dataset_size)
    buffer = ReplayBuffer(capacity=dataset_size)
    buffer.append_all(experiences)
    set_global_seed(args.seed)

    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        evaluator,
        timing=args.eval_timing,
        writer=W.FileWriter(outdir=outdir, file_prefix="evaluation_result"),
    )

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_num_hook = H.IterationNumHook(timing=100)
    iteration_state_hook = H.IterationStateHook(timing=100)

    config = A.IQLConfig(gpu_id=args.gpu)
    env_info = EnvironmentInfo.from_env(eval_env)
    iql = A.IQL(env_info, config=config, policy_solver_builder=CosineDecayPolicySolverBuilder(args.total_iterations))

    hooks = [save_snapshot_hook, evaluation_hook, iteration_num_hook, iteration_state_hook]
    iql.set_hooks(hooks)

    iql.train_offline(buffer, total_iterations=args.total_iterations)

    eval_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 200, render=args.render)
    config = A.BEARConfig(gpu_id=args.gpu)
    iql = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(iql, A.IQL):
        raise ValueError("Loaded snapshot is not trained with IQL!")

    evaluator = EpisodicEvaluator(args.showcase_runs)
    evaluator(iql, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah-medium-expert-v2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--total_iterations", type=int, default=1000000)
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
