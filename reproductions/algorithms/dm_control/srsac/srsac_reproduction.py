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

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_dmc_env, set_global_seed


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    eval_env = build_dmc_env(args.env, test=True, seed=args.seed + 100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=args.eval_timing,
                                       writer=W.FileWriter(outdir=outdir, file_prefix='evaluation_result'))

    iteration_num_hook = H.IterationNumHook(timing=100)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    train_env = build_dmc_env(args.env, seed=args.seed, render=args.render)
    if args.enable_rnn:
        config = A.SRSACConfig(gpu_id=args.gpu,
                               replay_ratio=args.replay_ratio,
                               reset_interval=args.reset_interval,
                               initial_temperature=1.0)
        srsac = A.SRSAC(train_env, config=config)

    else:
        config = A.EfficientSRSACConfig(gpu_id=args.gpu,
                                        replay_ratio=args.replay_ratio,
                                        reset_interval=args.reset_interval,
                                        initial_temperature=1.0)
        srsac = A.EfficientSRSAC(train_env, config=config)

    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    srsac.set_hooks(hooks)

    srsac.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    eval_env = build_dmc_env(args.env, test=True, seed=args.seed + 200, render=args.render)
    if args.enable_rnn:
        config = A.SRSACConfig(gpu_id=args.gpu)
        srsac = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
        if not isinstance(srsac, A.SRSAC):
            raise ValueError('Loaded snapshot is not trained with SRSAC!')
    else:
        config = A.EfficientSRSACConfig(gpu_id=args.gpu)
        srsac = serializers.load_snapshot(args.snapshot_dir, eval_env, algorithm_kwargs={"config": config})
        if not isinstance(srsac, A.EfficientSRSAC):
            raise ValueError('Loaded snapshot is not trained with EfficientSRSAC!')

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(srsac, eval_env)


def main():
    parser = argparse.ArgumentParser()
    choices = ['acrobot-swingup', 'cheetah-run', 'finger-turn_hard', 'fish-swim', 'hopper-hop',
               'hopper-stand', 'humanoid-run', 'humanoid-stand', 'humanoid-walk', 'pendulum-swingup',
               'quadruped-run', 'quadruped-walk', 'reacher-hard', 'swimmer-swimmer6', 'walker-run']
    parser.add_argument('--env', type=str, default='acrobot-swingup', help=f'DMC15 benchmark: {choices}')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=500000)
    parser.add_argument('--save_timing', type=int, default=5000)
    parser.add_argument('--eval_timing', type=int, default=5000)
    parser.add_argument('--showcase_runs', type=int, default=10)

    # SRSAC algorithm config
    parser.add_argument('--replay-ratio', type=int, default=32)
    parser.add_argument('--reset-interval', type=int, default=2560000)
    parser.add_argument('--enable-rnn', action='store_true', help='Turn this option true if you want to use rnn models')

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
