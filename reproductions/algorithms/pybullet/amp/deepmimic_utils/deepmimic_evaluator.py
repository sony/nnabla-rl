# Copyright 2024 Sony Group Corporation.
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

from typing import List, Union

from nnabla_rl.environments.amp_env import AMPEnv, AMPGoalEnv
from nnabla_rl.logger import logger
from nnabla_rl.utils.evaluator import run_one_episode


class DeepMimicEpisodicEvaluator:
    def __init__(self, run_per_evaluation: int = 32):
        self._num_episodes = run_per_evaluation

    def __call__(self, algorithm, env: Union[AMPEnv, AMPGoalEnv]) -> List[float]:
        returns: List[float] = []
        while len(returns) < self._num_episodes:
            reward_sum, timesteps, experiences = run_one_episode(algorithm, env)
            state, action, reward, non_terminal, next_state, info = experiences[-1]
            # the last element is always done
            assert not non_terminal
            # the info of last element should have valid_episode key
            assert "valid_episode" in info

            if info["valid_episode"]:
                returns.append(reward_sum)
                logger.info("Finished evaluation run: #{} out of {}. Total reward: {}".format(
                    len(returns), self._num_episodes, reward_sum))
            else:
                logger.info("Invalid episode. Skip to add this episode to evaluation")
        return returns
