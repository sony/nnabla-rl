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

import nnabla as nn
from nnabla_rl.algorithms.common_utils import (_StatePreprocessedPolicy, _StatePreprocessedVFunction,
                                               compute_v_target_and_advantage)
from nnabla_rl.models import VFunction


class DummyVFunction(VFunction):
    def __init__(self):
        super(DummyVFunction, self).__init__("test_v_function")
        self._state_dim = 1

    def v(self, s):
        with nn.parameter_scope(self.scope_name):
            h = s * 2.
        return h


class TestCommonUtils():
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

    def test_state_preprocessed_v_function(self):
        state_shape = (5, )

        from nnabla_rl.models import TRPOVFunction
        v_scope_name = 'old_v'
        v_function = TRPOVFunction(v_scope_name)

        import nnabla_rl.preprocessors as RP
        preprocessor_scope_name = 'test_preprocessor'
        preprocessor = RP.RunningMeanNormalizer(preprocessor_scope_name, shape=state_shape)

        v_function_old = _StatePreprocessedVFunction(v_function=v_function, preprocessor=preprocessor)

        s = nn.Variable.from_numpy_array(np.empty(shape=(1, *state_shape)))
        _ = v_function_old.v(s)

        v_new_scope_name = 'new_v'
        v_function_new = v_function_old.deepcopy(v_new_scope_name)

        assert v_function_old.scope_name != v_function_new.scope_name
        assert v_function_old._preprocessor.scope_name == v_function_new._preprocessor.scope_name

    def test_state_preprocessed_policy(self):
        state_shape = (5, )
        action_dim = 10

        from nnabla_rl.models import TRPOPolicy
        pi_scope_name = 'old_pi'
        pi = TRPOPolicy(pi_scope_name, action_dim=action_dim)

        import nnabla_rl.preprocessors as RP
        preprocessor_scope_name = 'test_preprocessor'
        preprocessor = RP.RunningMeanNormalizer(preprocessor_scope_name, shape=state_shape)

        pi_old = _StatePreprocessedPolicy(policy=pi, preprocessor=preprocessor)

        s = nn.Variable.from_numpy_array(np.empty(shape=(1, *state_shape)))
        _ = pi_old.pi(s)

        pi_new_scope_name = 'new_v'
        pi_new = pi_old.deepcopy(pi_new_scope_name)

        assert pi_old.scope_name != pi_new.scope_name
        assert pi_old._preprocessor.scope_name == pi_new._preprocessor.scope_name


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    pytest.main()
