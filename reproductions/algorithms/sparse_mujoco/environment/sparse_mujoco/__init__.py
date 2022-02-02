# Copyright 2022 Sony Group Corporation.
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

register(
    id='SparseHalfCheetah-v1',
    entry_point='sparse_mujoco.sparse_half_cheetah:SparseHalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SparseHopper-v1',
    entry_point='sparse_mujoco.sparse_hopper:SparseHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SparseWalker2d-v1',
    max_episode_steps=1000,
    entry_point='sparse_mujoco.sparse_walker2d:SparseWalker2dEnv',
)

register(
    id='SparseAnt-v1',
    entry_point='sparse_mujoco.sparse_ant:SparseAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
