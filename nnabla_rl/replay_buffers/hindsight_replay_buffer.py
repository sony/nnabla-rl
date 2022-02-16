# Copyright 2021,2022 Sony Group Corporation.
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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

import nnabla_rl as rl
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.typing import Experience


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self,
                 reward_function: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Any],
                 hindsight_prob: float = 0.8,
                 capacity: Optional[int] = None):
        super(HindsightReplayBuffer, self).__init__(capacity=capacity)
        self._reward_function = reward_function
        self._hindsight_prob = hindsight_prob
        self._current_episode_index = 0
        self._index_in_episode = 0
        self._start_index_of_episode = [0]
        self._index_in_episode = 0
        self._episode_end_index = np.array([0])  # workaround to share the value across episode

    def append(self, experience: Experience):
        # experience = (s, a, r, non_terminal, s_next, info)
        if not isinstance(experience[0], tuple):
            raise RuntimeError('Hindsight replay only supports tuple observation environment')
        if not len(experience[0]) == 3:
            raise RuntimeError('Observation is not a tuple of 3 elements: (observation, desired_goal, achieved_goal)')
        # Here, info will be updated.
        if not isinstance(experience[5], dict):
            raise ValueError
        non_terminal = experience[3]
        done = non_terminal == 0
        self._episode_end_index[0] = self._index_in_episode  # end index is shared among episode
        update_info = {'index_in_episode': self._index_in_episode,
                       'episode_end_index': self._episode_end_index}
        experience[5].update(update_info)

        super().append(experience)

        if done:
            self._index_in_episode = 0
            self._episode_end_index = np.array([0])  # workaround to share the value across episode
        else:
            self._index_in_episode += 1

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1) -> Tuple[Sequence[Experience], Dict[str, Any]]:
        # n-step learning is not supported
        if 1 < num_steps:
            raise NotImplementedError

        if len(indices) == 0:
            raise ValueError('Indices are empty')
        weights = np.ones([len(indices), 1])
        return [self._sample_experience(index) for index in indices], dict(weights=weights)

    def _sample_experience(self, index: int) -> Experience:
        if rl.random.drng.random() > self._hindsight_prob:
            # no change of experience
            return self.__getitem__(index)
        else:
            return self._make_hindsight_experience(index)

    def _make_hindsight_experience(self, index: int) -> Experience:
        # state = (observation, desired_goal, achieved_goal)
        experience = self.__getitem__(index)
        experience_info = experience[5]
        index_in_episode = experience_info['index_in_episode']
        episode_end_index = int(experience_info['episode_end_index'])  # NOTE: episode_end_index is saved as np.ndarray
        distance_to_end = episode_end_index - index_in_episode

        # sample index for hindsight goal
        episode_end_index = index + distance_to_end
        future_index = self._select_future_index(index, episode_end_index)

        # replace goal
        future_experience = self.__getitem__(future_index)
        new_experience = self._replace_goal(experience, future_experience)

        # save for test
        new_experience[-1].update({'future_index': future_index})

        return new_experience

    def _select_future_index(self, index_in_episode, episode_end_index):
        return rl.random.drng.integers(index_in_episode, min(episode_end_index + 1, self.capacity))

    def _replace_goal(self, current_experience: Experience, future_experience: Experience) -> Experience:
        s, a, _, non_terminal, s_next, info = current_experience
        future_s_next = future_experience[4]
        future_goal = future_s_next[2]
        new_s = (s[0], future_goal, s[2])
        new_s_next = (s_next[0], future_goal, s_next[2])
        new_r = self._reward_function(new_s_next[2], future_goal, info)
        return (new_s, a, new_r, non_terminal, new_s_next, info)
