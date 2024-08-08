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
import pathlib
from typing import Union

import gym
import numpy as np
from deepmimic_utils.deepmimic_buffer import RandomRemovalReplayBuffer
from deepmimic_utils.deepmimic_env import DeepMimicEnv, DeepMimicGoalEnv, DeepMimicWindowViewer
from deepmimic_utils.deepmimic_evaluator import DeepMimicEpisodicEvaluator
from deepmimic_utils.deepmimic_explorer import DeepMimicExplorer
from deepmimic_utils.deepmimic_normalizer import (
    DeepMimicGoalTupleRunningMeanNormalizer,
    DeepMimicTupleRunningMeanNormalizer,
)

import nnabla_rl.algorithms as A
import nnabla_rl.environment_explorers as EE
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.builders import ExplorerBuilder, PreprocessorBuilder, ReplayBufferBuilder
from nnabla_rl.environment_explorer import EnvironmentExplorer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environments.wrappers import FlattenNestedTupleStateWrapper, NumpyFloat32Env
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.utils import serializers
from nnabla_rl.utils.reproductions import print_env_info, set_global_seed


class DeepMimicTupleStatePreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(  # type: ignore[override]
        self,
        scope_name: str,
        env_info: EnvironmentInfo,
        algorithm_config: A.AMPConfig,
        **kwargs,
    ) -> Preprocessor:
        assert algorithm_config.state_mean_initializer is not None
        assert algorithm_config.state_var_initializer is not None

        if env_info.is_goal_conditioned_env():
            return DeepMimicGoalTupleRunningMeanNormalizer(
                scope_name,
                policy_state_shape=env_info.state_shape[0],
                reward_state_shape=env_info.state_shape[1],
                goal_state_shape=env_info.state_shape[3],
                policy_state_mean_initializer=np.array(algorithm_config.state_mean_initializer[0], dtype=np.float32),
                policy_state_var_initializer=np.array(algorithm_config.state_var_initializer[0], dtype=np.float32),
                reward_state_mean_initializer=np.array(algorithm_config.state_mean_initializer[1], dtype=np.float32),
                reward_state_var_initializer=np.array(algorithm_config.state_var_initializer[1], dtype=np.float32),
                goal_state_mean_initializer=np.array(algorithm_config.state_mean_initializer[3], dtype=np.float32),
                goal_state_var_initializer=np.array(algorithm_config.state_var_initializer[3], dtype=np.float32),
                epsilon=0.02,
                mode_for_floating_point_error="max",
            )
        else:
            return DeepMimicTupleRunningMeanNormalizer(
                scope_name,
                policy_state_shape=env_info.state_shape[0],
                reward_state_shape=env_info.state_shape[1],
                policy_state_mean_initializer=np.array(algorithm_config.state_mean_initializer[0], dtype=np.float32),
                policy_state_var_initializer=np.array(algorithm_config.state_var_initializer[0], dtype=np.float32),
                reward_state_mean_initializer=np.array(algorithm_config.state_mean_initializer[1], dtype=np.float32),
                reward_state_var_initializer=np.array(algorithm_config.state_var_initializer[1], dtype=np.float32),
                epsilon=0.02,
                mode_for_floating_point_error="max",
            )


class DeepMimicExplorerBuilder(ExplorerBuilder):
    def build_explorer(  # type: ignore[override]
        self,
        env_info: EnvironmentInfo,
        algorithm_config: A.AMPConfig,
        algorithm: A.AMP,
        **kwargs,
    ) -> EnvironmentExplorer:
        explorer_config = EE.LinearDecayEpsilonGreedyExplorerConfig(
            initial_step_num=0,
            timelimit_as_terminal=algorithm_config.timelimit_as_terminal,
            initial_epsilon=1.0,
            final_epsilon=algorithm_config.final_explore_rate,
            max_explore_steps=algorithm_config.max_explore_steps,
            append_explorer_info=True,
        )
        explorer = DeepMimicExplorer(
            greedy_action_selector=kwargs["greedy_action_selector"],
            random_action_selector=kwargs["random_action_selector"],
            env_info=env_info,
            config=explorer_config,
        )
        return explorer


class DeepMimicReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(  # type: ignore[override]
        self, env_info: EnvironmentInfo, algorithm_config: A.AMPConfig, **kwargs
    ) -> RandomRemovalReplayBuffer:
        return RandomRemovalReplayBuffer(
            capacity=int(np.ceil(algorithm_config.discriminator_agent_replay_buffer_size / algorithm_config.actor_num))
        )


def build_deepmimic_env(
    args_file_path: str,
    goal_conditioned_env: bool,
    seed: int,
    eval_mode: bool,
    print_env: bool,
    num_processes: int,
    render_env: bool,
) -> gym.Env:
    env: gym.Env
    if args_file_path == "FakeAMPNNablaRL-v1":
        # NOTE: FakeAMPNNablaRL-v1 is for the algorithm test.
        env = gym.make(args_file_path)
    else:
        if goal_conditioned_env:
            env = GoalConditionedTupleObservationEnv(
                DeepMimicGoalEnv(
                    args_file_path, eval_mode, num_processes=num_processes, step_until_action_needed=not render_env
                )
            )
            env = FlattenNestedTupleStateWrapper(env)
        else:
            env = DeepMimicEnv(
                args_file_path, eval_mode, num_processes=num_processes, step_until_action_needed=not render_env
            )

    # dummy reset for generating core
    env.reset()

    if print_env:
        print_env_info(env)

    env = NumpyFloat32Env(env)
    env.seed(seed)
    return env


def build_config(args, train_env: Union[DeepMimicEnv, DeepMimicGoalEnv]):
    if args.goal_conditioned_env:
        observation_mean = tuple([mean.tolist() for mean in train_env.unwrapped.observation_mean["observation"]])
        observation_var = tuple([var.tolist() for var in train_env.unwrapped.observation_var["observation"]])
        desired_goal_mean = tuple([mean.tolist() for mean in train_env.unwrapped.observation_mean["desired_goal"]])
        desired_goal_var = tuple([var.tolist() for var in train_env.unwrapped.observation_var["desired_goal"]])
        achieved_goal_mean = tuple([mean.tolist() for mean in train_env.unwrapped.observation_mean["achieved_goal"]])
        achieved_goal_var = tuple([var.tolist() for var in train_env.unwrapped.observation_var["achieved_goal"]])

        config = A.AMPConfig(
            gpu_id=args.gpu,
            seed=args.seed,
            normalize_action=True,
            preprocess_state=True,
            use_reward_from_env=True,
            gamma=0.99,
            action_mean=tuple(train_env.unwrapped.action_mean.tolist()),
            action_var=tuple(train_env.unwrapped.action_var.tolist()),
            state_mean_initializer=tuple([*observation_mean, *desired_goal_mean, *achieved_goal_mean]),
            state_var_initializer=tuple([*observation_var, *desired_goal_var, *achieved_goal_var]),
            value_at_task_fail=train_env.unwrapped.reward_at_task_fail / (1.0 - 0.99),
            value_at_task_success=train_env.unwrapped.reward_at_task_success / (1.0 - 0.99),
            target_value_clip=(
                train_env.unwrapped.reward_range[0] / (1.0 - 0.99),
                train_env.unwrapped.reward_range[1] / (1.0 - 0.99),
            ),
            v_function_learning_rate=2e-05,
            policy_learning_rate=4e-06,
            actor_num=args.actor_num,
            actor_timesteps=4096 // args.actor_num,
            max_explore_steps=200000000 // args.actor_num,
        )
    else:
        config = A.AMPConfig(
            gpu_id=args.gpu,
            seed=args.seed,
            normalize_action=True,
            preprocess_state=True,
            gamma=0.95,
            action_mean=tuple(train_env.unwrapped.action_mean.tolist()),
            action_var=tuple(train_env.unwrapped.action_var.tolist()),
            state_mean_initializer=tuple([mean.tolist() for mean in train_env.unwrapped.observation_mean]),
            state_var_initializer=tuple([var.tolist() for var in train_env.unwrapped.observation_var]),
            value_at_task_fail=train_env.unwrapped.reward_at_task_fail / (1.0 - 0.95),
            value_at_task_success=train_env.unwrapped.reward_at_task_success / (1.0 - 0.95),
            target_value_clip=(
                train_env.unwrapped.reward_range[0] / (1.0 - 0.95),
                train_env.unwrapped.reward_range[1] / (1.0 - 0.95),
            ),
            actor_num=args.actor_num,
            actor_timesteps=4096 // args.actor_num,
            max_explore_steps=200000000 // args.actor_num,
        )
    return config


def run_training(args):
    env_name = str(pathlib.Path(args.args_file_path).name).replace("_args.txt", "").replace("train_amp_", "")
    outdir = f"{env_name}_results/seed-{args.seed}"
    if args.save_dir:
        outdir = os.path.join(os.path.abspath(args.save_dir), outdir)
    set_global_seed(args.seed)

    train_env = build_deepmimic_env(
        args.args_file_path,
        goal_conditioned_env=args.goal_conditioned_env,
        seed=args.seed,
        eval_mode=False,
        print_env=True,
        num_processes=args.actor_num,
        render_env=False,
    )
    config = build_config(args, train_env)

    amp = A.AMP(
        train_env,
        config=config,
        env_explorer_builder=DeepMimicExplorerBuilder(),
        state_preprocessor_builder=DeepMimicTupleStatePreprocessorBuilder(),
        discriminator_replay_buffer_builder=DeepMimicReplayBufferBuilder(),
    )

    eval_env = build_deepmimic_env(
        args.args_file_path,
        goal_conditioned_env=args.goal_conditioned_env,
        seed=args.seed + 100,
        eval_mode=True,
        print_env=False,
        num_processes=1,
        render_env=False,
    )
    evaluator = DeepMimicEpisodicEvaluator(run_per_evaluation=32)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        evaluator,
        timing=args.eval_timing,
        writer=W.FileWriter(outdir=outdir, file_prefix="evaluation_result"),
    )
    iteration_state_hook = H.IterationStateHook(
        writer=W.FileWriter(outdir=outdir, file_prefix="iteration_state"), timing=args.iteration_state_timing
    )

    iteration_num_hook = H.IterationNumHook(timing=5000)
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=args.save_timing)

    hooks = [iteration_num_hook, iteration_state_hook, save_snapshot_hook, evaluation_hook]
    amp.set_hooks(hooks)
    amp.train_online(train_env, total_iterations=args.total_iterations)

    eval_env.close()
    train_env.close()


def run_showcase(args):
    if args.snapshot_dir is None:
        raise ValueError("Please specify the snapshot dir for showcasing")
    eval_env = build_deepmimic_env(
        args.args_file_path,
        goal_conditioned_env=args.goal_conditioned_env,
        seed=args.seed + 200,
        eval_mode=True,
        print_env=True,
        num_processes=1,
        render_env=args.render_in_showcase,
    )
    config = build_config(args, eval_env)
    amp = serializers.load_snapshot(
        args.snapshot_dir,
        eval_env,
        algorithm_kwargs={"config": config, "state_preprocessor_builder": DeepMimicTupleStatePreprocessorBuilder()},
    )
    if not isinstance(amp, A.AMP):
        raise ValueError("Loaded snapshot is not trained with AMP!")

    if args.render_in_showcase:
        viewer = DeepMimicWindowViewer(eval_env, amp.compute_eval_action)
        viewer.render(args.showcase_runs)
    else:
        evaluator = DeepMimicEpisodicEvaluator(run_per_evaluation=args.showcase_runs)
        evaluator(amp, eval_env)


def main():
    parser = argparse.ArgumentParser()
    script_dir = pathlib.Path(__file__).parent
    parser.add_argument(
        "--args_file_path",
        type=str,
        default=str(script_dir / "args" / "train_amp_humanoid3d_cartwheel_args.txt"),
    )
    parser.add_argument("--goal_conditioned_env", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--actor_num", type=int, default=16)
    parser.add_argument("--total_iterations", type=int, default=200000000)
    parser.add_argument("--save_timing", type=int, default=1000000)
    parser.add_argument("--eval_timing", type=int, default=500000)
    parser.add_argument("--iteration_state_timing", type=int, default=100000)
    parser.add_argument("--showcase_runs", type=int, default=32)
    parser.add_argument("--render_in_showcase", action="store_true")

    args = parser.parse_args()

    if args.showcase:
        run_showcase(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
