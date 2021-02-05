# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from typing import Callable, Dict, Tuple

from dataclasses import dataclass

import numpy as np

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerParam


@dataclass
class RawPolicyExplorerParam(EnvironmentExplorerParam):
    pass


class RawPolicyExplorer(EnvironmentExplorer):
    def __init__(self,
                 policy_action_selector: Callable[[np.array], Tuple[np.array, Dict]],
                 env_info: EnvironmentInfo,
                 params: RawPolicyExplorerParam = RawPolicyExplorerParam()):
        super().__init__(env_info, params)
        self._policy_action_selector = policy_action_selector

    def action(self, step, state):
        return self._policy_action_selector(state)
