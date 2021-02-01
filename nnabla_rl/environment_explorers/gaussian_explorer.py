from typing import Callable, Dict, Tuple

from dataclasses import dataclass

import numpy as np
import sys

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environment_explorer import EnvironmentExplorer, EnvironmentExplorerParam


@dataclass
class GaussianExplorerParam(EnvironmentExplorerParam):
    action_clip_low: float = sys.float_info.min
    action_clip_high: float = sys.float_info.max
    sigma: float = 1.0

    def __post_init__(self):
        self._assert_positive(self.sigma, 'sigma')


class GaussianExplorer(EnvironmentExplorer):
    def __init__(self,
                 policy_action_selector: Callable[[np.array], Tuple[np.array, Dict]],
                 env_info: EnvironmentInfo,
                 params: GaussianExplorerParam = GaussianExplorerParam()):
        super().__init__(env_info, params)
        self._policy_action_selector = policy_action_selector

    def action(self, step, state):
        (action, info) = self._policy_action_selector(state)
        return self._append_noise(action, self._params.action_clip_low, self._params.action_clip_high), info

    def _append_noise(self, action, low, high):
        noise = np.random.normal(loc=0.0, scale=self._params.sigma, size=action.shape).astype(np.float32)
        return np.clip(action + noise, low, high)
