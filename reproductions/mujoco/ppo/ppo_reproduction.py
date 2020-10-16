import argparse

import gym

import numpy as np

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
import nnabla_rl.utils.context as context
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.utils.reproductions import build_mujoco_env
from nnabla_rl.hook import as_hook
from nnabla_rl.logger import logger
from nnabla_rl.utils import serializers


def select_only_reset_if_truncated(env_name):
    if 'Swimmer' in env_name:
        reset_if_truncated = False
    else:
        reset_if_truncated = True
    return reset_if_truncated

def run_training(args):
    nnabla_rl.run_on_gpu(cuda_device_id=args.gpu)

    outdir = '{}_results'.format(args.env)

    eval_env = build_mujoco_env(
        args.env, test=True, seed=100)
    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluation_hook = H.EvaluationHook(eval_env,
                                       evaluator,
                                       timing=5000,
                                       writer=W.FileWriter(outdir=outdir,
                                                           file_prefix='evaluation_result'))

    iteration_num_hook = H.IterationNumHook(timing=1000)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=5000)

    total_iterations = 1000000

    train_env = build_mujoco_env(args.env, seed=1, render=args.render)
    if args.snapshot_dir is None:
        only_reset_if_truncated = select_only_reset_if_truncated(args.env)
        params = A.PPOParam(epsilon=0.2,
                            entropy_coefficient=0.0,
                            actor_timesteps=2048,
                            epochs=10,
                            batch_size=64,
                            learning_rate=3.0*1e-4,
                            actor_num=1,
                            decrease_alpha=False,
                            only_reset_if_truncated=only_reset_if_truncated)
        ppo = A.PPO(train_env, params=params)
    else:
        ppo = serializers.load_snapshot(args.snapshot_dir)
    hooks = [iteration_num_hook, save_snapshot_hook, evaluation_hook]
    ppo.set_hooks(hooks)

    ppo.train_online(train_env, total_iterations=total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    nnabla_rl.run_on_gpu(cuda_device_id=0)

    if args.snapshot_dir is None:
        raise ValueError(
            'Please specify the snapshot dir for showcasing')
    ppo = serializers.load_snapshot(args.snapshot_dir)
    if not isinstance(ppo, A.PPO):
        raise ValueError('Loaded snapshot is not trained with PPO!')

    eval_env = build_mujoco_env(
        args.env, test=True, seed=200, render=True)
    evaluator = EpisodicEvaluator()
    evaluator(ppo, eval_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--gpu', type=int, default=0)
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
