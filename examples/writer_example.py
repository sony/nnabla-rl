import gym
from nnabla.monitor import Monitor, MonitorSeries

import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
from nnabla_rl.writer import Writer
from nnabla_rl.environments.wrappers import NumpyFloat32Env
from nnabla_rl.utils.files import create_dir_if_not_exist

import os


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
    params = A.DDPGParam(start_timesteps=200)
    ddpg = A.DDPG(train_env, params=params)
    ddpg.set_hooks(hooks=[training_state_hook])

    ddpg.train(train_env, total_iterations=10000)

    train_env.close()


if __name__ == "__main__":
    main()
