import pytest

import numpy as np

import nnabla as nn

import nnabla_rl.models as M
from nnabla_rl.algorithms.common_utils import compute_v_target_and_advantage


class DummyVFunction(M.VFunction):
    def __init__(self):
        super(DummyVFunction, self).__init__("test_v_function")
        self._state_dim = 1

    def v(self, s):
        with nn.parameter_scope(self.scope_name):
            h = s * 2.
        return h


class TestComputeVtargetandAdvantage():
    def setup_method(self, method):
        nn.clear_parameters()

    def _collect_dummy_experince(self, num_episodes=1, episode_length=3):
        experience = []
        for _ in range(num_episodes):
            for i in range(episode_length):
                s_current = np.ones(1, )
                a = np.ones(1, )
                s_next = np.ones(1, )
                r = np.ones(1, )
                non_terminal = np.ones(1, )
                if i == episode_length-1:
                    non_terminal = 0
                experience.append((s_current, a, r, non_terminal, s_next))
        return experience

    @pytest.mark.parametrize("gamma, lmb, expected",
                             [[1., 0., np.array([[1.], [1.], [-1.]])],
                              [1., 1., np.array([[1.], [0.], [-1.]])],
                              [0.9, 0.7, np.array([[0.9071], [0.17], [-1.]])],
                              ])
    def test_compute(self, gamma, lmb, expected):
        dummy_v_function = DummyVFunction()
        dummy_experince = self._collect_dummy_experince()

        actual_vtarg, actual_adv = compute_v_target_and_advantage(
            dummy_v_function, dummy_experince, gamma, lmb)

        assert np.allclose(actual_adv, expected)
        assert np.allclose(actual_vtarg, expected + 2.)
