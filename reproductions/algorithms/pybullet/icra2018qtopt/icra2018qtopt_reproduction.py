# Copyright 2022 Sony Group Corporation.
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
import concurrent.futures
import multiprocessing
import os
import pathlib
import pickle
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import tqdm
from external_grasping_env.kuka_grasping_procedural_env import KukaGraspingProceduralEnv

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.environment_explorers.raw_policy_explorer import RawPolicyExplorer, RawPolicyExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environments.wrappers import Float32RewardEnv, HWCToCHWEnv, TimestepAsStateEnv
from nnabla_rl.logger import logger
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience, State
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import set_global_seed


class KukaDiverseObjectReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: Optional[int] = None):
        super().__init__(capacity)

    def _to_float32(self, array):
        return np.array(array, dtype=np.float32)

    def __getitem__(self, item: int) -> Experience:
        s, a, r, non_terminal, s_next, info = super().__getitem__(item)
        s = tuple(map(self._to_float32, s))
        s_next = tuple(map(self._to_float32, s_next))
        return (s, a, r, non_terminal, s_next, info)


def build_kuka_grasping_procedural_env(test: bool = False, render: bool = False) -> gym.Env:
    # See: https://github.com/google-research/google-research/blob/master/dql_grasping/configs/env_procedural/grasping_env.gin  # noqa
    env = KukaGraspingProceduralEnv(
        block_random=0.3,
        camera_random=0,
        simple_observations=False,
        continuous=True,
        remove_height_hack=True,
        render_mode="GUI" if render else "DIRECT",
        num_objects=5,
        max_num_training_models=100 if test else 900,
        target=False,
        test=test
    )
    env = Float32RewardEnv(env)
    env = HWCToCHWEnv(env)
    env = TimestepAsStateEnv(env)
    return env


def kuka_grasping_procedural_env_random_action_selector(
    s: State, height_hack_prob: float = 0.9, *, begin_of_episode: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    dx, dy, dz, da = np.random.uniform(low=-1.0, high=1.0, size=4)
    # NOTE: Downbiased for z-axis
    # See: https://goo.gl/hPS6ca
    if np.random.random() < height_hack_prob:
        dz = -1
    return np.array([dx, dy, dz, da], dtype=np.float32), {}


def explore_env(episode_num: int) -> Experience:
    env = build_kuka_grasping_procedural_env(test=False)
    env_info = EnvironmentInfo.from_env(env)
    explorer = RawPolicyExplorer(
        kuka_grasping_procedural_env_random_action_selector,
        env_info,
        RawPolicyExplorerConfig(),
    )
    logger.info(f"Collecting {episode_num}th episode ...")
    return explorer.rollout(env)


def collect_data(
    num_episodes: int = 10000,
    max_episode_steps: int = 15,
    multi_process: bool = False,
    ncpu: Optional[int] = None,
    replay_buffer_file_path: Union[str, pathlib.Path] = "./replay_buffer_data.pickle",
):
    replay_buffer = KukaDiverseObjectReplayBuffer(int(max_episode_steps * num_episodes))

    if multi_process:
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
            results = executor.map(explore_env, [i for i in range(num_episodes)])
    else:
        results = [explore_env(i) for i in range(num_episodes)]

    for experiences in tqdm.tqdm(results):
        replay_buffer.append_all(experiences)

    with open(replay_buffer_file_path, mode="wb") as f:
        pickle.dump(replay_buffer, f)

    return replay_buffer


def run_training(args):
    outdir = (
        f"KukaGraspingProceduralEnv_{args.num_collection_episodes}_results/seed-{args.seed}"
    )
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    eval_env = build_kuka_grasping_procedural_env(test=True, render=args.render)
    evaluator = EpisodicEvaluator(run_per_evaluation=50)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        evaluator,
        timing=args.eval_timing,
        writer=W.FileWriter(outdir=outdir, file_prefix="evaluation_result"),
    )

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_num_hook = H.IterationNumHook(timing=100)
    latest_state_hook = H.IterationStateHook(timing=100)

    if os.path.exists(args.replay_buffer_file_path):
        with open(args.replay_buffer_file_path, mode="rb") as f:
            replay_buffer = pickle.load(f)
            logger.info(f"Loaded replay buffer size: {len(replay_buffer)}")
    else:
        replay_buffer = collect_data(
            num_episodes=args.num_collection_episodes,
            multi_process=args.multi_process,
            ncpu=args.ncpu,
            replay_buffer_file_path=args.replay_buffer_file_path,
        )

    train_env = build_kuka_grasping_procedural_env(test=False, render=args.render)

    # NOTE: Downbiased for z-axis
    # See: https://github.com/google-research/google-research/blob/master/dql_grasping/policies.py#L298
    config = A.ICRA2018QtOptConfig(
        gpu_id=args.gpu, cem_initial_mean=(0.0, 0.0, -1.0, 0.0), batch_size=args.batch_size
    )
    icra2018qtopt = A.ICRA2018QtOpt(train_env, config=config)

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook, latest_state_hook]
    icra2018qtopt.set_hooks(hooks)

    icra2018qtopt.train_offline(replay_buffer, total_iterations=args.total_iterations)
    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_kuka_grasping_procedural_env(test=True, render=args.render)

    config = A.ICRA2018QtOptConfig(gpu_id=args.gpu)
    icra2018qtopt = serializers.load_snapshot(
        args.snapshot_dir, eval_env, algorithm_kwargs={"config": config}
    )
    if not isinstance(icra2018qtopt, A.ICRA2018QtOpt):
        raise ValueError("Loaded snapshot is not trained with ICRA2018QtOpt!")

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(icra2018qtopt, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_iterations", type=int, default=2000000)
    parser.add_argument("--save_timing", type=int, default=10000)
    parser.add_argument("--eval_timing", type=int, default=5000)
    parser.add_argument("--showcase_runs", type=int, default=10)
    parser.add_argument("--num_collection_episodes", type=int, default=1000000)
    parser.add_argument("--multi_process", action="store_true")
    parser.add_argument("--ncpu", type=int, default=None)
    parser.add_argument(
        "--replay_buffer_file_path", type=str, default="./replay_buffer_1m_data.pkl"
    )

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
