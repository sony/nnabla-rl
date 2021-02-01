import pytest

import numpy as np

from nnabla_rl.environment_explorers.gaussian_explorer import GaussianExplorer, GaussianExplorerParam


class TestRandomGaussianActionStrategy(object):
    @pytest.mark.parametrize('clip_low', np.arange(start=-1.0, stop=0.0, step=0.25))
    @pytest.mark.parametrize("clip_high", np.arange(start=0.0, stop=1.0, step=0.25))
    @pytest.mark.parametrize("sigma", np.arange(start=0.01, stop=5.0, step=1.0))
    def test_random_gaussian_action_selection(self, clip_low, clip_high, sigma):
        def policy_action_selector(state):
            return np.zeros(shape=state.shape), {'test': 'success'}
        params = GaussianExplorerParam(
            action_clip_low=clip_low,
            action_clip_high=clip_high,
            sigma=sigma
        )
        explorer = GaussianExplorer(
            env_info=None,
            policy_action_selector=policy_action_selector,
            params=params
        )

        steps = 1
        state = np.empty(shape=(1, 4))
        action, info = explorer.action(steps, state)

        assert np.all(clip_low <= action) and np.all(action <= clip_high)
        assert info['test'] == 'success'


if __name__ == '__main__':
    pytest.main()
