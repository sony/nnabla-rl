# Copyright 2023 Sony Group Corporation.
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

import gym

import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.algorithms import HyAR, HyARConfig
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.environments.wrappers.common import PrintEpisodeResultEnv
from nnabla_rl.environments.wrappers.hybrid_env import (FlattenActionWrapper, MergeBoxActionWrapper, RemoveStepWrapper,
                                                        ScaleActionWrapper, ScaleStateWrapper)
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import print_env_info, set_global_seed

try:
    import gym_goal  # noqa
    from goal_env_wrapper import ExtendedGoalEnvWrapper
except ModuleNotFoundError:
    pass
try:
    import gym_platform  # noqa
except ModuleNotFoundError:
    pass


def setup_platform_env(env):
    env = FlattenActionWrapper(env)
    env = ScaleStateWrapper(env)
    env = ScaleActionWrapper(env)
    env = MergeBoxActionWrapper(env)
    env = RemoveStepWrapper(env)
    return env


def setup_goal_env(env):
    env = ExtendedGoalEnvWrapper(env)

    env = FlattenActionWrapper(env)
    env = ScaleStateWrapper(env)
    env = ScaleActionWrapper(env)
    env = MergeBoxActionWrapper(env)
    env = RemoveStepWrapper(env)
    return env


def build_env(env_name, test=False, seed=None, render=False, print_episode_result=False):
    env = gym.make(env_name)
    if env_name == 'Goal-v0':
        env = setup_goal_env(env)
    elif env_name == "Platform-v0":
        env = setup_platform_env(env)
    else:
        pass
    print_env_info(env)

    env = NumpyFloat32Env(env)
    if render:
        env = ScreenRenderEnv(env)
    if print_episode_result:
        env = PrintEpisodeResultEnv(env)
    env.seed(seed)

    return env


def setup_hyar(env, args):
    config = HyARConfig(gpu_id=args.gpu,
                        learning_rate=3e-4,
                        batch_size=128,
                        start_timesteps=128,
                        train_action_noise_abs=1.0,
                        train_action_noise_sigma=0.1,
                        replay_buffer_size=int(1e5),
                        vae_learning_rate=1e-4,
                        vae_pretrain_episodes=args.vae_pretrain_episodes,
                        vae_pretrain_times=args.vae_pretrain_times)
    return HyAR(env, config=config)


def setup_algorithm(env, args):
    return setup_hyar(env, args)


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    eval_env = build_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=100)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=args.eval_timing,
                                       writer=W.FileWriter(outdir=outdir, file_prefix='evaluation_result'))

    iteration_num_hook = H.IterationNumHook(timing=100)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    train_env = build_env(args.env, seed=args.seed, render=args.render)
    algorithm = setup_algorithm(train_env, args)

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    algorithm.set_hooks(hooks)
    algorithm.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    eval_env = build_env(args.env, seed=args.seed + 200, render=args.render)
    config = HyARConfig(gpu_id=args.gpu)
    hyar = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
    if not isinstance(hyar, HyAR):
        raise ValueError('Loaded snapshot is not trained with PPO!')

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(hyar, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Goal-v0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=300000)
    parser.add_argument('--save_timing', type=int, default=5000)
    parser.add_argument('--eval_timing', type=int, default=5000)
    parser.add_argument('--showcase_runs', type=int, default=10)
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--vae-pretrain-episodes', type=int, default=20000)
    parser.add_argument('--vae-pretrain-times', type=int, default=5000)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
