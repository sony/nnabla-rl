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

import numpy as np
import pytest

from nnabla_rl.environment_explorers.gaussian_explorer import GaussianExplorer, GaussianExplorerConfig


class TestRandomGaussianActionStrategy(object):
    @pytest.mark.parametrize('clip_low', np.arange(start=-1.0, stop=0.0, step=0.25))
    @pytest.mark.parametrize("clip_high", np.arange(start=0.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("sigma", np.arange(start=0.01, stop=5.0, step=1.0))
    def test_random_gaussian_action_selection(self, clip_low, clip_high, sigma):
        def policy_action_selector(state):
            return np.zeros(shape=state.shape), {'test': 'success'}
        config = GaussianExplorerConfig(
            action_clip_low=clip_low,
            action_clip_high=clip_high,
            sigma=sigma
        )
        explorer = GaussianExplorer(
            env_info=None,
            policy_action_selector=policy_action_selector,
            config=config
        )

        steps = 1
        state = np.empty(shape=(1, 4))
        action, info = explorer.action(steps, state)

        assert np.all(clip_low <= action) and np.all(action <= clip_high)
        assert info['test'] == 'success'


if __name__ == '__main__':
    pytest.main()
