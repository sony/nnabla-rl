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
