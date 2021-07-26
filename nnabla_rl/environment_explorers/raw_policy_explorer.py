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

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig
from nnabla_rl.environments.environment_info import EnvironmentInfo


@dataclass
class RawPolicyExplorerConfig(EnvironmentExplorerConfig):
    pass


class RawPolicyExplorer(EnvironmentExplorer):
    '''Raw policy explorer

    Explore using policy's action without any changes.

    Args:
        policy_action_selector (Callable[[np.ndarray], Tuple[np.ndarray, Dict]]):
            callable which computes current policy's action with respect to current state.
        env_info (:py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            environment info
        config (:py:class:`LinearDecayEpsilonGreedyExplorerConfig\
            <nnabla_rl.environment_explorers.RawPolicyExplorerConfig>`): the config of this class.
    '''

    def __init__(self,
                 policy_action_selector: Callable[[np.ndarray], Tuple[np.ndarray, Dict]],
                 env_info: EnvironmentInfo,
                 config: RawPolicyExplorerConfig = RawPolicyExplorerConfig()):
        super().__init__(env_info, config)
        self._policy_action_selector = policy_action_selector

    def action(self, step, state):
        return self._policy_action_selector(state)
