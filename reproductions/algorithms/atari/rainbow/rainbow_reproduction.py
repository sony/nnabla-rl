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

import nnabla_rl.algorithms as A
import nnabla_rl.environment_explorers as EE
import nnabla_rl.hooks as H
import nnabla_rl.replay_buffers as RB
import nnabla_rl.writers as W
from nnabla_rl.algorithms import RainbowConfig
from nnabla_rl.builders import ExplorerBuilder, ModelBuilder, ReplayBufferBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import (RainbowNoDuelValueDistributionFunction, RainbowNoNoisyValueDistributionFunction,
                              ValueDistributionFunction)
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator, TimestepEvaluator
from nnabla_rl.utils.reproductions import build_atari_env, set_global_seed


class MemoryEfficientPrioritizedBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self, env_info, algorithm_config, **kwargs):
        # Some of hyper-parameters was taken from: https://github.com/deepmind/dqn_zoo
        return RB.ProportionalPrioritizedAtariBuffer(capacity=algorithm_config.replay_buffer_size,
                                                     alpha=algorithm_config.alpha,
                                                     beta=algorithm_config.beta,
                                                     betasteps=algorithm_config.betasteps,
                                                     error_clip=(-100, 100),
                                                     normalization_method="batch_max")


class MemoryEfficientNonPrioritizedBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self, env_info, algorithm_config, **kwargs):
        return RB.MemoryEfficientAtariBuffer(capacity=algorithm_config.replay_buffer_size)


class NoDuelValueDistributionFunctionBuilder(ModelBuilder[ValueDistributionFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: RainbowConfig,
                    **kwargs) -> ValueDistributionFunction:
        return RainbowNoDuelValueDistributionFunction(scope_name,
                                                      env_info.action_dim,
                                                      algorithm_config.num_atoms,
                                                      algorithm_config.v_min,
                                                      algorithm_config.v_max)


class NoNoisyValueDistributionFunctionBuilder(ModelBuilder[ValueDistributionFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: RainbowConfig,
                    **kwargs) -> ValueDistributionFunction:
        return RainbowNoNoisyValueDistributionFunction(scope_name,
                                                       env_info.action_dim,
                                                       algorithm_config.num_atoms,
                                                       algorithm_config.v_min,
                                                       algorithm_config.v_max)


class EpsilonGreedyExplorerBuilder(ExplorerBuilder):
    def build_explorer(self, env_info, algorithm_config, algorithm, **kwargs):
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            warmup_random_steps=algorithm_config.warmup_random_steps,
            initial_step_num=algorithm.iteration_num,
            initial_epsilon=algorithm_config.initial_epsilon,
            final_epsilon=algorithm_config.final_epsilon,
            max_explore_steps=algorithm_config.max_explore_steps
        )
        explorer = EE.LinearDecayEpsilonGreedyExplorer(
            greedy_action_selector=algorithm._greedy_action_selector,
            random_action_selector=algorithm._random_action_selector,
            env_info=env_info,
            config=explorer_config)
        return explorer


def setup_no_double_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu, no_double=True)
    return A.Rainbow(train_env, config=config, replay_buffer_builder=MemoryEfficientPrioritizedBufferBuilder())


def setup_no_prior_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu)
    return A.Rainbow(train_env,
                     config=config,
                     replay_buffer_builder=MemoryEfficientNonPrioritizedBufferBuilder())


def setup_no_duel_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu)
    return A.Rainbow(train_env,
                     config=config,
                     value_distribution_builder=NoDuelValueDistributionFunctionBuilder(),
                     replay_buffer_builder=MemoryEfficientPrioritizedBufferBuilder())


def setup_no_n_steps_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu, num_steps=1)
    return A.Rainbow(train_env,
                     config=config,
                     replay_buffer_builder=MemoryEfficientPrioritizedBufferBuilder())


def setup_no_noisy_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu,
                             initial_epsilon=1.0,
                             final_epsilon=0.01,
                             test_epsilon=0.001,
                             max_explore_steps=250000 // 4)
    return A.Rainbow(train_env,
                     config=config,
                     value_distribution_builder=NoNoisyValueDistributionFunctionBuilder(),
                     replay_buffer_builder=MemoryEfficientPrioritizedBufferBuilder(),
                     explorer_builder=EpsilonGreedyExplorerBuilder())


def setup_full_rainbow(train_env, args):
    config = A.RainbowConfig(gpu_id=args.gpu)
    return A.Rainbow(train_env, config=config, replay_buffer_builder=MemoryEfficientPrioritizedBufferBuilder())


def suffix_from_algorithm_options(args):
    if args.no_double:
        return "-no_double"
    elif args.no_prior:
        return "-no_prior"
    elif args.no_duel:
        return "-no_duel"
    elif args.no_n_steps:
        return "-no_n_steps"
    elif args.no_noisy:
        return "-no_noisy"
    else:
        return ""


def setup_rainbow(train_env, args):
    if args.no_double:
        return setup_no_double_rainbow(train_env, args)
    elif args.no_prior:
        return setup_no_prior_rainbow(train_env, args)
    elif args.no_duel:
        return setup_no_duel_rainbow(train_env, args)
    elif args.no_n_steps:
        return setup_no_n_steps_rainbow(train_env, args)
    elif args.no_noisy:
        return setup_no_noisy_rainbow(train_env, args)
    else:
        return setup_full_rainbow(train_env, args)


def load_rainbow(env, args):
    if args.no_double:
        raise NotImplementedError
    elif args.no_prior:
        raise NotImplementedError
    elif args.no_duel:
        raise NotImplementedError
    elif args.no_n_steps:
        raise NotImplementedError
    elif args.no_noisy:
        raise NotImplementedError
    else:
        config = A.RainbowConfig(gpu_id=args.gpu)
        return serializers.load_snapshot(args.snapshot_dir, env, algorithm_kwargs={"config": config})


def run_training(args):
    suffix = suffix_from_algorithm_options(args)
    outdir = f'{args.env}{suffix}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    max_frames_per_episode = 30 * 60 * 60  # 30 min * 60 seconds * 60 fps
    eval_env = build_atari_env(args.env,
                               test=True, seed=args.seed + 100,
                               render=args.render,
                               max_frames_per_episode=max_frames_per_episode)
    evaluator = TimestepEvaluator(num_timesteps=125000)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=args.eval_timing,
                                       writer=W.FileWriter(outdir=outdir, file_prefix='evaluation_result'))
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)
    iteration_num_hook = H.IterationNumHook(timing=100)

    train_env = build_atari_env(args.env, seed=args.seed, render=args.render,
                                max_frames_per_episode=max_frames_per_episode)

    rainbow = setup_rainbow(train_env, args)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    rainbow.set_hooks(hooks)

    print(f'current Rainbow config: {rainbow._config}')
    rainbow.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    max_frames_per_episode = 30 * 60 * 60  # 30 min * 60 seconds * 60 fps
    eval_env = build_atari_env(args.env,
                               test=True,
                               seed=args.seed + 200,
                               render=args.render,
                               max_frames_per_episode=max_frames_per_episode)
    rainbow = load_rainbow(eval_env, args)
    if not isinstance(rainbow, A.Rainbow):
        raise ValueError('Loaded snapshot is not trained with Rainbow!')

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    evaluator(rainbow, eval_env)


def add_algorithm_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--no-double', action='store_true')
    group.add_argument('--no-prior', action='store_true')
    group.add_argument('--no-n-steps', action='store_true')
    group.add_argument('--no-noisy', action='store_true')
    group.add_argument('--no-duel', action='store_true')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--save-dir', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--total_iterations', type=int, default=50000000)
    parser.add_argument('--save_timing', type=int, default=250000)
    parser.add_argument('--eval_timing', type=int, default=250000)
    parser.add_argument('--showcase_runs', type=int, default=10)
    add_algorithm_options(parser)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
