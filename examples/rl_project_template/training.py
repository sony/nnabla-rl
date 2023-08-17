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
import pathlib

from environment import TemplateEnv
from models import TemplatePolicy, TemplateQFunction

import nnabla_rl.hooks as H
from nnabla_rl.algorithm import AlgorithmConfig
from nnabla_rl.algorithms import SAC, SACConfig
from nnabla_rl.builders import ModelBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.utils.reproductions import set_global_seed


class QFunctionBuilder(ModelBuilder):
    def build_model(self, scope_name: str, env_info: EnvironmentInfo, algorithm_config: AlgorithmConfig, **kwargs):
        return TemplateQFunction(scope_name)


class PolicyBuilder(ModelBuilder):
    def build_model(self, scope_name: str, env_info: EnvironmentInfo, algorithm_config: AlgorithmConfig, **kwargs):
        return TemplatePolicy(scope_name, env_info.action_dim)


def create_env(seed):
    env = TemplateEnv()
    env.seed(seed)
    return env


def build_algorithm(env, args):
    # Select a RL algorithm of your choice and build its instance.
    # In this sample, we selected Soft-Actor Critic(SAC).

    # Setting the configuration of the algorithm.
    config = SACConfig(gpu_id=args.gpu)

    # Algorithm requires not only the configuration but also the target models to train.
    algorithm = SAC(env, config, q_function_builder=QFunctionBuilder(), policy_builder=PolicyBuilder())

    return algorithm


def run_training(args):
    # It is optional but we set the random seed to ensure reproducibility.
    set_global_seed(args.seed)

    # Create two different environments to avoid using training environment in the evaluation.
    train_env = create_env(args.seed)
    eval_env = create_env(args.seed + 100)
    algorithm = build_algorithm(train_env, args)

    # nnabla-rl has a convenient feature that enables running additional operations
    # in each specified timing (iteration).
    # This hook evaluates the training model every "timing" iteration steps.
    evaluation_hook = H.EvaluationHook(eval_env, timing=1000)

    # Adding this hook to just check that the training runs properly.
    # This hook prints current iteration number every "timing" (=100) steps.
    iteration_num_hook = H.IterationNumHook(timing=100)

    # Save trained parameters every "timing" steps.
    # Without this, the parameters will not be saved.
    # We recommend saving parameters at every evaluation timing.
    outdir = pathlib.Path(args.save_dir) / 'snapshots'
    save_snapshot_hook = H.SaveSnapshotHook(outdir=outdir, timing=1000)

    # All instantiated hooks should be set at once.
    # set_hooks will override previously set hooks.
    algorithm.set_hooks([evaluation_hook, iteration_num_hook, save_snapshot_hook])

    algorithm.train(train_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save-dir', type=str, default=str(pathlib.Path(__file__).parent))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    main()
