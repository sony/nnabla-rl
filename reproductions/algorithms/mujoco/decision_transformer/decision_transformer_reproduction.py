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
from typing import cast

import gym
import numpy as np

import nnabla as nn
import nnabla.solvers as NS
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla.utils.learning_rate_scheduler import BaseLearningRateScheduler
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.builders.lr_scheduler_builder import LearningRateSchedulerBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.logger import logger
from nnabla_rl.models import MujocoDecisionTransformer
from nnabla_rl.replay_buffers import TrajectoryReplayBuffer
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env, set_global_seed
from nnabla_rl.utils.solver_wrappers import AutoClipGradByNorm
from nnabla_rl.writers import FileWriter

try:
    # import at the end. d4rl overrides logger unexpectedly
    import d4rl  # noqa
except ModuleNotFoundError:
    # Ignore if d4rl is not installed
    pass


class StateNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, state_mean, state_std):
        gym.ObservationWrapper.__init__(self, env)
        self._state_mean = state_mean
        self._state_std = state_std

    def observation(self, state):
        return (state - self._state_mean) / self._state_std


class MujocoLearningRateScheduler(BaseLearningRateScheduler):
    def __init__(self, initial_learning_rate, warmup_steps):
        super().__init__()
        self._initial_learning_rate = initial_learning_rate
        self._warmup_steps = warmup_steps
        self._step_num = 0

    def get_learning_rate(self, iter):
        self._step_num += 1
        new_learning_rate = self._initial_learning_rate * min(self._step_num / self._warmup_steps, 1.0)
        return new_learning_rate


class MujocoLearningRateSchedulerBuilder(LearningRateSchedulerBuilder):
    def __init__(self, warmup_steps) -> None:
        super().__init__()
        self._warmup_steps = warmup_steps

    def build_scheduler(self, env_info, algorithm_config, **kwargs) -> BaseLearningRateScheduler:
        return MujocoLearningRateScheduler(algorithm_config.learning_rate, self._warmup_steps)


class MujocoDecisionTransformerBuilder(ModelBuilder):
    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        max_timesteps = cast(int, kwargs['max_timesteps'])
        return MujocoDecisionTransformer(scope_name,
                                         env_info.action_dim,
                                         max_timestep=max_timesteps,
                                         context_length=algorithm_config.context_length)


class MujocoSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs) -> nn.solver.Solver:
        # Set initial alpha used internally in AdamW to 1.0
        solver = NS.AdamW(alpha=1.0, wd=algorithm_config.weight_decay)
        # Set true learning rate here
        solver.set_learning_rate(algorithm_config.learning_rate)
        return AutoClipGradByNorm(solver, algorithm_config.grad_clip_norm)


def load_d4rl_dataset(env_name, dataset_type):
    if 'HalfCheetah' in env_name:
        task_name = 'halfcheetah'
    elif 'Hopper' in env_name:
        task_name = 'hopper'
    elif 'Walker2d' in env_name:
        task_name = 'walker2d'
    d4rl_name = f'{task_name}-{dataset_type}-v2'
    d4rl_env = gym.make(d4rl_name)
    return d4rl_env.get_dataset()


def load_dataset_from_path(dataset_dir):
    import gzip
    import pathlib

    def load_data_from_gz(gzfile):
        with gzip.open(gzfile, mode='rb') as f:
            data = np.load(f, allow_pickle=False)
        return data

    dataset = {}

    dataset_dir = pathlib.Path(dataset_dir)
    observation_file = dataset_dir / '$store$_observation_ckpt.0.gz'
    action_file = dataset_dir / '$store$_action_ckpt.0.gz'
    reward_file = dataset_dir / '$store$_reward_ckpt.0.gz'
    terminal_file = dataset_dir / '$store$_terminal_ckpt.0.gz'
    next_observation_file = dataset_dir / '$store$_next_observation_ckpt.0.gz'

    observations = load_data_from_gz(observation_file)
    actions = load_data_from_gz(action_file)
    rewards = load_data_from_gz(reward_file)
    terminals = load_data_from_gz(terminal_file)
    next_observations = load_data_from_gz(next_observation_file)

    dataset['observations'] = observations
    dataset['actions'] = actions
    dataset['rewards'] = rewards
    dataset['terminals'] = terminals
    dataset['next_observations'] = next_observations

    return dataset


def compute_state_mean_and_std(d4rl_dataset):
    state_mean = np.mean(d4rl_dataset['observations'], axis=0)
    state_std = np.std(d4rl_dataset['observations'], axis=0) + 1e-6

    return state_mean, state_std


def load_dataset(d4rl_dataset, buffer_size, context_length, reward_scale):
    use_timeouts = 'timeouts' in d4rl_dataset

    max_possible_trajectories = buffer_size // context_length
    buffer = TrajectoryReplayBuffer(num_trajectories=max_possible_trajectories)

    dataset_size = d4rl_dataset['rewards'].shape[0]

    max_timesteps = 1
    episode_step = 0
    start_index = 0
    state_mean, state_std = compute_state_mean_and_std(d4rl_dataset)
    for i in range(dataset_size):
        done = bool(d4rl_dataset['terminals'][i])
        episode_step = i - start_index
        final_timestep = d4rl_dataset['timeouts'][i] if use_timeouts else (episode_step == 1000 - 1)
        if done or final_timestep:
            end_index = i
            states = (d4rl_dataset['observations'][start_index:end_index+1] - state_mean) / state_std
            actions = d4rl_dataset['actions'][start_index:end_index+1]
            rewards = d4rl_dataset['rewards'][start_index:end_index+1] * reward_scale
            non_terminals = 1.0 - d4rl_dataset['terminals'][start_index:end_index+1]
            next_states = (d4rl_dataset['next_observations'][start_index:end_index+1] - state_mean) / state_std

            start_index = end_index + 1

            info = [{} for _ in range(len(states))]
            for timestep, d in enumerate(info):
                d['rtg'] = np.sum(rewards[timestep:])
                d['timesteps'] = timestep
            assert all([len(data) == len(states) for data in (actions, rewards, non_terminals, next_states, info)])
            timesteps = len(info) - 1
            trajectory = list(zip(states, actions, rewards, non_terminals, next_states, info))

            buffer.append_trajectory(trajectory)
            max_timesteps = max(max_timesteps, timesteps)
    return buffer, max_timesteps


def get_target_return(env_name):
    if 'HalfCheetah' in env_name:
        return 6000
    if 'Hopper' in env_name:
        return 3600
    if 'Walker' in env_name:
        return 5000
    raise NotImplementedError(f'No target_return is defined for: {env_name}')


def get_reward_scale(env_name):
    if 'HalfCheetah' in env_name:
        return 1/1000
    if 'Hopper' in env_name:
        return 1/1000
    if 'Walker' in env_name:
        return 1/1000
    return 1.0


def get_context_length(env_name):
    return 20


def run_training(args):
    outdir = f'{args.env}_results/seed-{args.seed}'
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    context_length = args.context_length if args.context_length is not None else get_context_length(args.env)
    reward_scale = args.reward_scale if args.reward_scale is not None else get_reward_scale(args.env)
    if args.dataset_path is None:
        d4rl_dataset = load_d4rl_dataset(args.env, args.dataset_type)
    else:
        d4rl_dataset = load_dataset_from_path(args.dataset_path)
    dataset, max_timesteps = load_dataset(d4rl_dataset, args.buffer_size, context_length, reward_scale)
    state_mean, state_std = compute_state_mean_and_std(d4rl_dataset)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 100, render=args.render)
    eval_env = StateNormalizationWrapper(eval_env, state_mean, state_std)

    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env, evaluator, timing=args.eval_timing, writer=writer)

    epoch_num_hook = H.EpochNumHook(iteration_per_epoch=1)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    target_return = args.target_return if args.target_return is not None else get_target_return(args.env)
    config = A.DecisionTransformerConfig(gpu_id=args.gpu,
                                         context_length=context_length,
                                         max_timesteps=max_timesteps,
                                         batch_size=args.batch_size,
                                         target_return=target_return,
                                         grad_clip_norm=0.25,
                                         learning_rate=1.0e-4,
                                         weight_decay=1.0e-4,
                                         reward_scale=reward_scale)
    env_info = EnvironmentInfo.from_env(eval_env)
    decision_transformer = A.DecisionTransformer(
        env_info,
        config=config,
        transformer_builder=MujocoDecisionTransformerBuilder(),
        transformer_solver_builder=MujocoSolverBuilder(),
        transformer_wd_solver_builder=None,
        lr_scheduler_builder=MujocoLearningRateSchedulerBuilder(args.warmup_steps))
    decision_transformer.set_hooks(hooks=[epoch_num_hook, save_snapshot_hook, evaluation_hook])

    print(f'total epochs: {args.total_epochs}')
    # decision transformer runs 1 epoch per iteration
    decision_transformer.train(dataset, total_iterations=args.total_epochs)

    eval_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError('Please specify the snapshot dir for showcasing')
    if args.dataset_path is None:
        dataset = load_d4rl_dataset(args.env, args.dataset_type)
    else:
        dataset = load_dataset_from_path(args.dataset_path)
    state_mean, state_std = compute_state_mean_and_std(dataset)

    eval_env = build_mujoco_env(args.env, test=True, seed=args.seed + 200, render=args.render)
    eval_env = StateNormalizationWrapper(eval_env, state_mean, state_std)
    config = {'gpu_id': args.gpu}
    decision_transformer = serializers.load_snapshot(
        args.snapshot_dir,
        eval_env,
        algorithm_kwargs={"config": config, "transformer_builder": MujocoDecisionTransformerBuilder()})
    if not isinstance(decision_transformer, A.DecisionTransformer):
        raise ValueError('Loaded snapshot is not trained with DecisionTransformer!')

    evaluator = EpisodicEvaluator(run_per_evaluation=args.showcase_runs)
    returns = evaluator(decision_transformer, eval_env)
    mean = np.mean(returns)
    std_dev = np.std(returns)
    median = np.median(returns)
    logger.info('Evaluation results. mean: {} +/- std: {}, median: {}'.format(mean, std_dev, median))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--dataset-type', type=str, default='medium', choices=['medium', 'expert'])
    parser.add_argument('--save-dir', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--total-epochs', type=int, default=5)
    parser.add_argument('--trajectories-per-buffer', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=10000)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--context-length', type=int, default=None)
    parser.add_argument('--save_timing', type=int, default=1)
    parser.add_argument('--eval_timing', type=int, default=1)
    parser.add_argument('--showcase_runs', type=int, default=10)
    parser.add_argument('--target-return', type=int, default=None)
    parser.add_argument('--reward-scale', type=float, default=None)

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
