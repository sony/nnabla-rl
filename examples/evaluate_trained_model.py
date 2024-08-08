# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

import gym

from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.utils import serializers
from nnabla_rl.utils.evaluator import EpisodicEvaluator


def build_env():
    try:
        env = gym.make("Pendulum-v0")
    except gym.error.DeprecatedEnv:
        env = gym.make("Pendulum-v1")
    env = NumpyFloat32Env(env)
    env = ScreenRenderEnv(env)
    return env


def main():
    snapshot_dir = "./pendulum_v0_snapshot/iteration-10000"
    env = build_env()

    algorithm = serializers.load_snapshot(snapshot_dir, env)

    evaluator = EpisodicEvaluator(run_per_evaluation=10)
    evaluator(algorithm, env)

    env.close()


if __name__ == "__main__":
    main()
