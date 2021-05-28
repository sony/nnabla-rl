# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from gym.envs.registration import register

from nnabla_rl.environments.dummy import (DummyAtariEnv, DummyContinuous, DummyDiscrete,  # noqa
                                          DummyDiscreteImg, DummyMujocoEnv)

register(
    id='FakeMujocoNNablaRL-v1',
    entry_point='nnabla_rl.environments.dummy:DummyMujocoEnv',
    max_episode_steps=10
)

register(
    id='FakeAtariNNablaRLNoFrameskip-v1',
    entry_point='nnabla_rl.environments.dummy:DummyAtariEnv',
    max_episode_steps=10
)
