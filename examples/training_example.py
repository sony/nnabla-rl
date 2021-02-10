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

import gym

import nnabla_rl
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv


def build_env(seed=None):
    env = gym.make('Pendulum-v0')
    env = NumpyFloat32Env(env)
    env = ScreenRenderEnv(env)
    env.seed(seed)
    return env


def main():
    # Run on gpu if possible.
    nnabla_rl.run_on_gpu(0)

    # Evaluate the trained network (Optional)
    eval_env = build_env(seed=100)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        timing=1000,
        writer=W.FileWriter(outdir='./pendulum_v0_ddpg_results', file_prefix='evaluation_result'))

    # Pring iteration number every 100 iterations.
    iteration_num_hook = H.IterationNumHook(timing=100)

    # Save trained algorithm snapshot (Optional)
    save_snapshot_hook = H.SaveSnapshotHook('./pendulum_v0_ddpg_results', timing=1000)

    train_env = build_env()
    config = A.DDPGConfig(start_timesteps=200)
    ddpg = A.DDPG(train_env, config=config)
    ddpg.set_hooks(hooks=[iteration_num_hook, save_snapshot_hook, evaluation_hook])

    ddpg.train(train_env, total_iterations=10000)

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
