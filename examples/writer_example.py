# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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

import os

import gym

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla.monitor import Monitor, MonitorSeries
from nnabla_rl.environments.wrappers import NumpyFloat32Env
from nnabla_rl.utils.files import create_dir_if_not_exist
from nnabla_rl.writer import Writer


class MyScalarWriter(Writer):
    def __init__(self, outdir):
        self._outdir = os.path.join(outdir, 'writer')
        create_dir_if_not_exist(outdir=self._outdir)
        self._monitor = Monitor(self._outdir)
        self._monitor_series = None
        super().__init__()

    def write_scalar(self, iteration_num, scalar):
        if self._monitor_series is None:
            self._create_monitor_series(scalar.keys())

        for writer, value in zip(self._monitor_series, scalar.values()):
            writer.add(iteration_num, value)

    def _create_monitor_series(self, names):
        self._monitor_series = []
        for name in names:
            self._monitor_series.append(MonitorSeries(
                name, self._monitor, interval=1, verbose=False))


def build_env(seed=None):
    env = gym.make('Pendulum-v0')
    env = NumpyFloat32Env(env)
    env.seed(seed)
    return env


def main():
    writer = MyScalarWriter('./pendulum_v0_ddpg_results')
    training_state_hook = H.IterationStateHook(writer=writer, timing=100)

    train_env = build_env()
    config = A.DDPGConfig(start_timesteps=200)
    ddpg = A.DDPG(train_env, config=config)
    ddpg.set_hooks(hooks=[training_state_hook])

    ddpg.train(train_env, total_iterations=10000)

    train_env.close()


if __name__ == "__main__":
    main()
