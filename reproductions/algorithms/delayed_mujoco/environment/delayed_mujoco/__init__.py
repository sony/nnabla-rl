# Copyright 2022,2023,2024 Sony Group Corporation.
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
    id="DelayedHalfCheetah-v1",
    entry_point="delayed_mujoco.delayed_mujoco:DelayedHalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="DelayedHopper-v1",
    entry_point="delayed_mujoco.delayed_mujoco:DelayedHopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="DelayedWalker2d-v1",
    max_episode_steps=1000,
    entry_point="delayed_mujoco.delayed_mujoco:DelayedWalker2dEnv",
)

register(
    id="DelayedAnt-v1",
    entry_point="delayed_mujoco.delayed_mujoco:DelayedAntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
