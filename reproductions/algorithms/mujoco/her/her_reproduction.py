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
from typing import List

import gym

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.typing import Experience
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicSuccessEvaluator
from nnabla_rl.utils.reproductions import print_env_info, set_global_seed

gamma = 0.98
tau = 0.05
exploration_epsilon = 0.3
exploration_noise_sigma = 0.2
n_rollout = 16
n_update = 40  # the number of updating model in each cycle
start_timesteps = 1
batch_size = 256


def select_n_cycles(env_name):
    if env_name in ['FetchReach-v1']:
        n_cycles = 10
    else:
        n_cycles = 50
    print(f'Selected start n_cycles: {n_cycles}')
    return n_cycles


def check_success(experiences: List[Experience]) -> bool:
    last_info = experiences[-1][-1]
    if last_info['is_success'] == 1.0:
        return True
    else:
        return False


def build_mujoco_goal_conditioned_env(id_or_env, test=False, seed=None, render=False):
    try:
        # Add pybullet env
        import pybullet_envs  # noqa
    except ModuleNotFoundError:
        # Ignore if pybullet is not installed
        pass
    try:
        # Add d4rl env
        import d4rl  # noqa
    except ModuleNotFoundError:
        # Ignore if d4rl is not installed
        pass

    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = gym.make(id_or_env)
    env = GoalConditionedTupleObservationEnv(env)
    print_env_info(env)

    env = NumpyFloat32Env(env)

    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    n_cycles = select_n_cycles(env_name=args.env)
    train_env = build_mujoco_goal_conditioned_env(args.env, seed=args.seed, render=args.render)
    eval_env = build_mujoco_goal_conditioned_env(args.env, test=True, seed=args.seed + 100, render=args.render)

    max_timesteps = train_env.spec.max_episode_steps
    iteration_per_epoch = n_cycles * n_update * max_timesteps

    evaluator = EpisodicSuccessEvaluator(check_success=check_success, run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator,
                                       timing=iteration_per_epoch,
                                       writer=W.FileWriter(outdir=outdir, file_prefix='evaluation_result'))

    epoch_num_hook = H.EpochNumHook(iteration_per_epoch=iteration_per_epoch)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    return_clip_min = -1. / (1. - gamma)
    return_clip_max = 0.0
    return_clip = (return_clip_min, return_clip_max)
    config = A.HERConfig(gpu_id=args.gpu,
                         gamma=gamma,
                         tau=tau,
                         exploration_noise_sigma=exploration_noise_sigma,
                         max_timesteps=max_timesteps,
                         n_cycles=n_cycles,
                         n_rollout=n_rollout,
                         n_update=n_update,
                         start_timesteps=start_timesteps,
                         batch_size=batch_size,
                         exploration_epsilon=exploration_epsilon,
                         return_clip=return_clip)
    her = A.HER(train_env, config=config)

    hooks = [epoch_num_hook, save_snapshot_hook, evaluation_hook]
    her.set_hooks(hooks)

    her.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    eval_env = build_mujoco_goal_conditioned_env(args.env, test=True, seed=args.seed + 200, render=args.render)
    config = A.HERConfig(gpu_id=args.gpu)
    her = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(her, A.HER):
        raise ValueError('Loaded snapshot is not trained with HER!')
    evaluator = EpisodicSuccessEvaluator(check_success=check_success,
                                         run_per_evaluation=args.showcase_runs)
    evaluator(her, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchPush-v1')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--save_timing', type=int, default=100000)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=20000000)
    parser.add_argument('--showcase_runs', type=int, default=10)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
