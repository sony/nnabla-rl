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

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.hook import Hook
from nnabla_rl.replay_buffer import ReplayBuffer


class PrintHello(Hook):
    def __init__(self):
        super().__init__(timing=1)

    def on_hook_called(self, algorithm):
        print("hello!!")


class PrintOnlyEvenIteraion(Hook):
    def __init__(self):
        super().__init__(timing=2)

    def on_hook_called(self, algorithm):
        print("even iteration -> {}".format(algorithm.iteration_num))


def main():
    dummy_env = E.DummyContinuous()
    empty_buffer = ReplayBuffer()

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[PrintHello()])
    dummy_algorithm.train(empty_buffer, total_iterations=10)

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[PrintHello(), PrintOnlyEvenIteraion()])
    dummy_algorithm.train(empty_buffer, total_iterations=10)


if __name__ == "__main__":
    main()
