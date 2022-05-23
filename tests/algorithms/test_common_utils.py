# Copyright 2020,2021 Sony Corporation.
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

import numpy as np
import pytest

import nnabla as nn
from nnabla_rl.algorithms.common_utils import (_DeterministicPolicyActionSelector,
                                               _StatePreprocessedDeterministicPolicy,
                                               _StatePreprocessedStochasticPolicy, _StatePreprocessedVFunction,
                                               compute_average_v_target_and_advantage, compute_v_target_and_advantage,
                                               has_batch_dimension)
from nnabla_rl.environments.dummy import (DummyContinuous, DummyContinuousActionGoalEnv, DummyDiscrete,
                                          DummyTupleContinuous, DummyTupleDiscrete)
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import VFunction


class DummyVFunction(VFunction):
    def __init__(self):
        super(DummyVFunction, self).__init__("test_v_function")

    def v(self, s):
        with nn.parameter_scope(self.scope_name):
            if isinstance(s, tuple):
                h = s[0] * 2. + s[1] * 2.
            else:
                h = s * 2.
        return h


class TestCommonUtils():
    def setup_method(self, method):
        nn.clear_parameters()

    def _collect_dummy_experience(self, num_episodes=1, episode_length=3, tupled_state=False):
        experience = []
        for _ in range(num_episodes):
            for i in range(episode_length):
                s_current = (np.ones(1, ), np.ones(1, )) if tupled_state else np.ones(1, )
                a = np.ones(1, )
                s_next = (np.ones(1, ), np.ones(1, )) if tupled_state else np.ones(1, )
                r = np.ones(1, )
                non_terminal = np.ones(1, )
                if i == episode_length-1:
                    non_terminal = 0
                experience.append((s_current, a, r, non_terminal, s_next))
        return experience

    def test_has_batch_dimension_tupled_continuous_state(self):
        env = DummyTupleContinuous()
        env_info = EnvironmentInfo.from_env(env)

        batch_size = 5
        state_shapes = env_info.state_shape
        batched_state = tuple(np.empty(shape=(batch_size, *state_shape)) for state_shape in state_shapes)
        non_batched_state = tuple(np.empty(shape=state_shape) for state_shape in state_shapes)

        assert has_batch_dimension(batched_state, env_info)
        assert not has_batch_dimension(non_batched_state, env_info)

    def test_has_batch_dimension_tupled_discrete_state(self):
        env = DummyTupleDiscrete()
        env_info = EnvironmentInfo.from_env(env)

        batch_size = 5
        state_shapes = env_info.state_shape
        batched_state = tuple(np.empty(shape=(batch_size, *state_shape)) for state_shape in state_shapes)
        non_batched_state = tuple(np.empty(shape=state_shape) for state_shape in state_shapes)

        assert has_batch_dimension(batched_state, env_info)
        assert not has_batch_dimension(non_batched_state, env_info)

    def test_has_batch_dimension_non_tupled_continuous_state(self):
        env = DummyContinuous()
        env_info = EnvironmentInfo.from_env(env)

        batch_size = 5
        state_shape = env_info.state_shape
        batched_state = np.empty(shape=(batch_size, *state_shape))
        non_batched_state = np.empty(shape=state_shape)

        assert has_batch_dimension(batched_state, env_info)
        assert not has_batch_dimension(non_batched_state, env_info)

    def test_has_batch_dimension_non_tupled_discrete_state(self):
        env = DummyDiscrete()
        env_info = EnvironmentInfo.from_env(env)

        batch_size = 5
        state_shape = env_info.state_shape
        batched_state = np.empty(shape=(batch_size, *state_shape))
        non_batched_state = np.empty(shape=state_shape)

        assert has_batch_dimension(batched_state, env_info)
        assert not has_batch_dimension(non_batched_state, env_info)

    @pytest.mark.parametrize("gamma, lmb, expected_adv, expected_vtarg, tupled_state",
                             [[1., 0., np.array([[1.], [1.], [-1.]]), np.array([[3.], [3.], [1.]]), False],
                              [1., 1., np.array([[1.], [0.], [-1.]]), np.array([[3.], [2.], [1.]]), False],
                              [0.9, 0.7, np.array([[0.9071], [0.17], [-1.]]),
                               np.array([[2.9071], [2.17], [1.]]), False],
                              [1., 0., np.array([[1.], [1.], [-3.]]), np.array([[5.], [5.], [1.]]), True],
                              [1., 1., np.array([[-1.], [-2.], [-3.]]), np.array([[3.], [2.], [1.]]), True],
                              ])
    def test_compute_v_target_and_advantage(self, gamma, lmb, expected_adv, expected_vtarg, tupled_state):
        dummy_v_function = DummyVFunction()
        dummy_experience = self._collect_dummy_experience(tupled_state=tupled_state)

        actual_vtarg, actual_adv = compute_v_target_and_advantage(
            dummy_v_function, dummy_experience, gamma, lmb)

        assert np.allclose(actual_adv, expected_adv)
        assert np.allclose(actual_vtarg, expected_vtarg)

    @pytest.mark.parametrize("lmb, expected_adv, expected_vtarg, tupled_state",
                             [[0., np.array([[0.], [0.], [-2.]]), np.array([[2.], [2.], [0.]]), False],
                              [1., np.array([[-2.], [-2.], [-2.]]), np.array([[0.], [0.], [0.]]), False],
                              [0.7, np.array([[-0.98], [-1.4], [-2.]]), np.array([[1.02], [0.6], [0.]]), False],
                              [0., np.array([[0.], [0.], [-4.]]), np.array([[4.], [4.], [0.]]), True],
                              [1., np.array([[-4.], [-4.], [-4.]]), np.array([[0.], [0.], [0.]]), True],
                              ])
    def test_compute_average_v_target_and_advantage(self, lmb, expected_adv, expected_vtarg, tupled_state):
        dummy_v_function = DummyVFunction()
        dummy_experience = self._collect_dummy_experience(tupled_state=tupled_state)

        actual_vtarg, actual_adv = compute_average_v_target_and_advantage(
            dummy_v_function, dummy_experience, lmb)

        assert np.allclose(actual_adv, expected_adv)
        assert np.allclose(actual_vtarg, expected_vtarg)

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

    def test_state_preprocessed_stochastic_policy(self):
        state_shape = (5, )
        action_dim = 10

        from nnabla_rl.models import TRPOPolicy
        pi_scope_name = 'old_pi'
        pi = TRPOPolicy(pi_scope_name, action_dim=action_dim)

        import nnabla_rl.preprocessors as RP
        preprocessor_scope_name = 'test_preprocessor'
        preprocessor = RP.RunningMeanNormalizer(preprocessor_scope_name, shape=state_shape)

        pi_old = _StatePreprocessedStochasticPolicy(policy=pi, preprocessor=preprocessor)

        s = nn.Variable.from_numpy_array(np.empty(shape=(1, *state_shape)))
        _ = pi_old.pi(s)

        pi_new_scope_name = 'new_pi'
        pi_new = pi_old.deepcopy(pi_new_scope_name)

        assert pi_old.scope_name != pi_new.scope_name
        assert pi_old._preprocessor.scope_name == pi_new._preprocessor.scope_name

    def test_state_preprocessed_deterministic_policy(self):
        state_shape = (5, )
        action_dim = 10

        from nnabla_rl.models import TD3Policy
        pi_scope_name = 'old_pi'
        pi = TD3Policy(pi_scope_name, action_dim=action_dim, max_action_value=1.0)

        import nnabla_rl.preprocessors as RP
        preprocessor_scope_name = 'test_preprocessor'
        preprocessor = RP.RunningMeanNormalizer(preprocessor_scope_name, shape=state_shape)

        pi_old = _StatePreprocessedDeterministicPolicy(policy=pi, preprocessor=preprocessor)

        s = nn.Variable.from_numpy_array(np.empty(shape=(1, *state_shape)))
        _ = pi_old.pi(s)

        pi_new_scope_name = 'new_pi'
        pi_new = pi_old.deepcopy(pi_new_scope_name)

        assert pi_old.scope_name != pi_new.scope_name
        assert pi_old._preprocessor.scope_name == pi_new._preprocessor.scope_name

    def test_action_selector_tupled_state(self):
        from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv

        env = DummyContinuousActionGoalEnv()
        env = GoalConditionedTupleObservationEnv(env)
        env_info = EnvironmentInfo.from_env(env)

        action_dim = env_info.action_dim

        from nnabla_rl.models import HERPolicy
        pi_scope_name = 'pi'
        pi = HERPolicy(pi_scope_name, action_dim=action_dim, max_action_value=1.0)

        selector = _DeterministicPolicyActionSelector(env_info, pi)

        batch_size = 5
        state_shapes = env_info.state_shape
        batched_state = tuple(np.empty(shape=(batch_size, *state_shape)) for state_shape in state_shapes)

        action, *_ = selector(batched_state)
        assert len(action) == batch_size
        assert action.shape[1:] == env_info.action_shape

        non_batched_state = tuple(np.empty(shape=state_shape) for state_shape in state_shapes)
        action, *_ = selector(non_batched_state)

        assert action.shape == env_info.action_shape

    def test_action_selector_non_tupled_state(self):
        env = DummyContinuous()
        env_info = EnvironmentInfo.from_env(env)

        action_dim = env_info.action_dim

        from nnabla_rl.models import TD3Policy
        pi_scope_name = 'pi'
        pi = TD3Policy(pi_scope_name, action_dim=action_dim, max_action_value=1.0)

        selector = _DeterministicPolicyActionSelector(env_info, pi)

        batch_size = 5
        state_shape = env_info.state_shape
        batched_state = np.empty(shape=(batch_size, *state_shape))

        action, *_ = selector(batched_state)
        assert len(action) == batch_size
        assert action.shape[1:] == env_info.action_shape

        non_batched_state = np.empty(shape=state_shape)
        action, *_ = selector(non_batched_state)

        assert action.shape == env_info.action_shape


if __name__ == "__main__":
    pytest.main()
