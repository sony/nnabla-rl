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

from gym.envs.mujoco.walker2d import Walker2dEnv


class SparseWalker2dEnv(Walker2dEnv):
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        reward = 1.0 if posafter - self.init_qpos[0] > 1.0 else 0.0

        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}
