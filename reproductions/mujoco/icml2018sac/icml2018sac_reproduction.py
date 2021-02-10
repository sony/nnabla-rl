# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed
from nnabla_rl.utils import serializers


def select_start_timesteps(env_name):
    if env_name in ['Ant-v2', 'HalfCheetah-v2']:
        timesteps = 10000
    else:
        timesteps = 1000
    print(f'Selected start timesteps: {timesteps}')
    return timesteps


def select_total_iterations(env_name):
    if env_name in ['Hopper-v2', 'Walker2d-v2']:
        total_iterations = 1000000
    elif env_name in ['Humanoid-v2']:
        total_iterations = 10000000
    else:
        total_iterations = 3000000
    print(f'Selected total iterations: {total_iterations}')
    return total_iterations


def select_reward_scalar(env_name):
    if env_name in ['Humanoid-v2']:
        scalar = 20.0
    else:
        scalar = 5.0
    print(f'Selected reward scalar: {scalar}')
    return scalar


def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = f'{args.env}_results/seed-{args.seed}'
    set_global_seed(args.seed)

    eval_env = build_mujoco_env(
        args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=5000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=5000)
    iteration_num_hook = H.IterationNumHook(timing=100)

    train_env = build_mujoco_env(args.env, seed=args.seed, render=args.render)
    if args.snapshot_dir is None:
        timesteps = select_start_timesteps(args.env)
        reward_scalar = select_reward_scalar(args.env)
        config = A.ICML2018SACConfig(
            start_timesteps=timesteps, reward_scalar=reward_scalar)
        icml2018sac = A.ICML2018SAC(train_env, config=config)
    else:
        icml2018sac = serializers.load_snapshot(args.snapshot_dir)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    icml2018sac.set_hooks(hooks)

    total_iterations = select_total_iterations(args.env)
    icml2018sac.train_online(train_env, total_iterations=total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    icml2018sac = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(icml2018sac, A.ICML2018SAC):
        raise ValueError('Loaded snapshot is not trained with ICML2018SAC!')

    eval_env = build_mujoco_env(
        args.env, test=True, seed=args.seed + 200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(icml2018sac, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
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
