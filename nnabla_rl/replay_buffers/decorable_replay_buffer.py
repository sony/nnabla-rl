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

from nnabla_rl.replay_buffer import ReplayBuffer


class DecorableReplayBuffer(ReplayBuffer):
    '''Buffer which can decorate the experience with external decoration function

    This buffer enables decorating the experience before the item is used for building the batch.
    Decoration function will be called when __getitem__ is called.
    You can use this buffer to augment the data or add noise to the experience.
    '''

    def __init__(self, capacity, decor_fun):
        super(DecorableReplayBuffer, self).__init__(capacity=capacity)
        self._decor_fun = decor_fun

    def __getitem__(self, item):
        experience = self._buffer[item]
        return self._decor_fun(experience)
