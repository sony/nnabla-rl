# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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
from packaging.version import parse

import nnabla as nn
import nnabla_rl.environments as E
from nnabla_rl.utils.data import (RingBuffer, add_batch_dimension, compute_std_ndarray, list_of_dict_to_dict_of_list,
                                  marshal_dict_experiences, marshal_experiences, normalize_ndarray,
                                  set_data_to_variable, unnormalize_ndarray)


class TestData():
    def test_set_data_to_variable(self):
        variable = nn.Variable((3,))
        array = np.random.rand(3)

        set_data_to_variable(variable, array)

        assert np.allclose(variable.d, array)

    def test_set_data_to_variable_tuple(self):
        variables = (nn.Variable((3,)), nn.Variable((5,)))
        arrays = (np.random.rand(3), np.random.rand(5))

        set_data_to_variable(variables, arrays)

        assert np.allclose(variables[0].d, arrays[0])
        assert np.allclose(variables[1].d, arrays[1])

    def test_set_data_to_variable_wrong_length(self):
        variables = (nn.Variable((3,)), nn.Variable((5,)), nn.Variable((7,)))
        arrays = (np.random.rand(3), np.random.rand(5))

        with pytest.raises(AssertionError):
            set_data_to_variable(variables, arrays)

    def test_marshal_experiences(self):
        batch_size = 3
        dummy_env = E.DummyContinuous()
        experiences = generate_dummy_experiences(dummy_env, batch_size)
        state, action, reward, done, next_state, info = marshal_experiences(experiences)
        rnn_states = info['rnn_states']
        rnn_dummy_state1 = rnn_states['dummy_scope']['dummy_state1']
        rnn_dummy_state2 = rnn_states['dummy_scope']['dummy_state2']

        assert state.shape == (batch_size, dummy_env.observation_space.shape[0])
        assert action.shape == (batch_size, dummy_env.action_space.shape[0])
        assert reward.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)
        assert next_state.shape == (batch_size, dummy_env.observation_space.shape[0])
        assert rnn_dummy_state1.shape == (batch_size, 1)
        assert rnn_dummy_state2.shape == (batch_size, 1)

    def test_marshal_experiences_tuple_continous(self):
        batch_size = 2
        dummy_env = E.DummyTupleContinuous()
        experiences = generate_dummy_experiences(dummy_env, batch_size)
        state, action, reward, done, next_state, info = marshal_experiences(experiences)
        rnn_states = info['rnn_states']
        rnn_dummy_state1 = rnn_states['dummy_scope']['dummy_state1']
        rnn_dummy_state2 = rnn_states['dummy_scope']['dummy_state2']

        assert state[0].shape == (batch_size, dummy_env.observation_space[0].shape[0])
        assert state[1].shape == (batch_size, dummy_env.observation_space[1].shape[0])
        assert action[0].shape == (batch_size, dummy_env.action_space[0].shape[0])
        assert action[1].shape == (batch_size, dummy_env.action_space[1].shape[0])
        assert reward.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)
        assert next_state[0].shape == (batch_size, dummy_env.observation_space[0].shape[0])
        assert next_state[1].shape == (batch_size, dummy_env.observation_space[1].shape[0])
        assert rnn_dummy_state1.shape == (batch_size, 1)
        assert rnn_dummy_state2.shape == (batch_size, 1)

    def test_marshal_experiences_tuple_discrete(self):
        batch_size = 2
        dummy_env = E.DummyTupleDiscrete()
        experiences = generate_dummy_experiences(dummy_env, batch_size)
        state, action, reward, done, next_state, info = marshal_experiences(experiences)
        rnn_states = info['rnn_states']
        rnn_dummy_state1 = rnn_states['dummy_scope']['dummy_state1']
        rnn_dummy_state2 = rnn_states['dummy_scope']['dummy_state2']

        assert state[0].shape == (batch_size, 1)
        assert state[1].shape == (batch_size, 1)
        assert action[0].shape == (batch_size, 1)
        assert action[1].shape == (batch_size, 1)
        assert reward.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)
        assert next_state[0].shape == (batch_size, 1)
        assert next_state[1].shape == (batch_size, 1)
        assert rnn_dummy_state1.shape == (batch_size, 1)
        assert rnn_dummy_state2.shape == (batch_size, 1)

    def test_marshal_dict_experiences(self):
        experiences = {'key1': 1, 'key2': 2}
        dict_experiences = [{'key_parent': experiences}, {'key_parent': experiences}]
        marshaled_experience = marshal_dict_experiences(dict_experiences)

        key1_experiences = marshaled_experience['key_parent']['key1']
        key2_experiences = marshaled_experience['key_parent']['key2']

        assert key1_experiences.shape == (2, 1)
        assert key2_experiences.shape == (2, 1)

        np.testing.assert_allclose(np.asarray(key1_experiences), 1)
        np.testing.assert_allclose(np.asarray(key2_experiences), 2)

    def test_marshal_triple_nested_dict_experiences(self):
        experiences = {'key1': 1, 'key2': 2}
        nested_experiences = {'nest1': experiences, 'nest2': experiences}
        dict_experiences = [{'key_parent': nested_experiences}, {'key_parent': nested_experiences}]
        marshaled_experience = marshal_dict_experiences(dict_experiences)

        key1_experiences = marshaled_experience['key_parent']['nest1']['key1']
        key2_experiences = marshaled_experience['key_parent']['nest2']['key2']

        assert len(key1_experiences) == 2
        assert len(key2_experiences) == 2

        np.testing.assert_allclose(np.asarray(key1_experiences), 1)
        np.testing.assert_allclose(np.asarray(key2_experiences), 2)

    def test_marashal_dict_experiences_with_inhomogeneous_part(self):
        installed_numpy_version = parse(np.__version__)
        numpy_version1_24 = parse('1.24.0')

        if installed_numpy_version < numpy_version1_24:
            # no need to test
            return

        experiences = {'key1': 1, 'key2': 2}
        inhomgeneous_experiences = {'key1': np.empty(shape=(6, )), 'key2': 2}
        dict_experiences = [{'key_parent': experiences}, {'key_parent': inhomgeneous_experiences}]

        marshaled_experience = marshal_dict_experiences(dict_experiences)

        assert 'key1' not in marshaled_experience['key_parent']

        key2_experiences = marshaled_experience['key_parent']['key2']
        assert key2_experiences.shape == (2, 1)

        np.testing.assert_allclose(np.asarray(key2_experiences), 2)

    def test_list_of_dict_to_dict_of_list(self):
        list_of_dict = [{'key1': 1, 'key2': 2}, {'key1': 1, 'key2': 2}]
        dict_of_list = list_of_dict_to_dict_of_list(list_of_dict)

        key1_list = dict_of_list['key1']
        key2_list = dict_of_list['key2']

        assert len(key1_list) == 2
        assert len(key2_list) == 2

        np.testing.assert_allclose(np.asarray(key1_list), 1)
        np.testing.assert_allclose(np.asarray(key2_list), 2)

    def test_add_batch_dimension_array(self):
        array = np.random.randn(4)
        actual_array = add_batch_dimension(array)

        assert actual_array.shape == (1, *array.shape)

    def test_add_batch_dimension_tuple(self):
        array1 = np.random.randn(4)
        array2 = np.random.randn(3)

        actual_array = add_batch_dimension((array1, array2))

        assert actual_array[0].shape == (1, *array1.shape)
        assert actual_array[1].shape == (1, *array2.shape)

    @pytest.mark.parametrize("x, mean, std, value_clip, expected",
                             [
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]), None,  np.array([5.0])),
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]),  (-1.5, 1.5), np.array([1.5])),
                                 (np.array([-2.0]), np.array([1.0]), np.array([0.2]),  (-1.5, 1.5), np.array([-1.5])),
                                 (np.array([[2.0], [1.0]]), np.array([[1.0]]), np.array([[0.2]]), None,
                                  np.array([[5.0], [0.0]])),
                             ])
    def test_normalize_ndarray(self, x, expected, mean, std, value_clip):
        actual_var = normalize_ndarray(x, mean, std, value_clip=value_clip)
        assert np.allclose(actual_var, expected)

    @pytest.mark.parametrize("x, mean, std, value_clip, expected",
                             [
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]), None, np.array([1.4])),
                                 (np.array([2.0]), np.array([1.0]), np.array([0.2]),  (-1.0, 1.0), np.array([1.0])),
                                 (np.array([-2.0]), np.array([-1.0]), np.array([0.2]),  (-1.0, 1.0),  np.array([-1.0])),
                                 (np.array([[2.0], [1.0]]), np.array([[1.0]]), np.array([[0.2]]), None,
                                  np.array([[1.4], [1.2]])),
                             ])
    def test_unnormalize_ndarray(self, x, expected, mean, std, value_clip):
        actual_var = unnormalize_ndarray(x, mean, std, value_clip=value_clip)
        assert np.allclose(actual_var, expected)

    @pytest.mark.parametrize("var, epsilon, mode_for_floating_point_error, expected",
                             [
                                 (np.array([3.0]), 1.0, "add", np.array([2.0])),
                                 (np.array([4.0]), 0.01, "max", np.array([2.0])),
                                 (np.array([0.4]), 1.0, "max", np.array([1.0])),
                                 (np.array([[3.0], [8.0]]), 1.0, "add", np.array([[2.0], [3.0]])),
                                 (np.array([[4.0], [9.0]]), 0.01, "max", np.array([[2.0], [3.0]])),
                                 (np.array([[0.4], [0.9]]), 1.0, "max", np.array([[1.0], [1.0]])),
                             ])
    def test_compute_std_ndarray(self, var, epsilon, mode_for_floating_point_error, expected):
        actual_var = compute_std_ndarray(var, epsilon, mode_for_floating_point_error)
        assert np.allclose(actual_var, expected)

    def test_compute_std_ndarray_with_invalid_args(self):
        with pytest.raises(ValueError):
            compute_std_ndarray(np.ones(1), 0.01, "dummy_add")


class TestRingBuffer(object):
    def test_append(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            assert len(buffer) == i
            buffer.append(i)
        assert len(buffer) == maxlen

        for i in range(maxlen):
            assert len(buffer) == maxlen
            buffer.append(i)
        assert len(buffer) == maxlen

    def test_getitem(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            buffer.append(i)
            assert i == buffer[i]

        for i in range(maxlen):
            buffer.append(i + maxlen)
            assert i + 1 == buffer[0]
            assert i + maxlen == buffer[maxlen - 1]

    def test_buffer_len(self):
        maxlen = 10
        buffer = RingBuffer(maxlen)
        for i in range(maxlen):
            assert len(buffer) == i
            buffer.append(i)
        assert len(buffer) == maxlen

        for i in range(maxlen):
            assert len(buffer) == maxlen
            buffer.append(i)
        assert len(buffer) == maxlen


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
