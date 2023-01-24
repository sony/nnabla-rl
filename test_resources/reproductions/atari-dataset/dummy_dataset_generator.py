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

import gzip

import numpy as np

from nnabla_rl.utils.reproductions import build_atari_env


def save_dataset(filepath, data):
    with gzip.GzipFile(filepath, 'w') as f:
        np.save(f, data, allow_pickle=False)


def main():
    fake_env = build_atari_env('FakeAtariNNablaRLNoFrameskip-v1', test=True)

    dataset_size = 20

    states = []
    actions = []
    rewards = []
    dones = []

    state = fake_env.reset()
    for t in range(dataset_size):
        action = fake_env.action_space.sample()
        next_state, reward, done, *_ = fake_env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        # Force final state as terminal
        dones.append(t % 2 == 1)

        state = fake_env.reset() if done else next_state

    save_dataset('./$store$_observation_ckpt.0.gz', np.asarray(states))
    save_dataset('./$store$_action_ckpt.0.gz', np.asarray(actions))
    save_dataset('./$store$_reward_ckpt.0.gz', np.asarray(rewards))
    save_dataset('./$store$_terminal_ckpt.0.gz', np.asarray(dones))


if __name__ == '__main__':
    main()
