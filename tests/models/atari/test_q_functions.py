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

from unittest import mock

import numpy as np

import nnabla as nn
from nnabla_rl.models.atari.q_functions import DQNQFunction


class TestDQNQFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        input_action = nn.Variable.from_numpy_array(np.ones((1, 1)))
        model.q(input_state, input_action)

        assert len(model.get_parameters()) == 10

    def test_q(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        input_action = nn.Variable.from_numpy_array(np.ones((1, 1)))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model.all_q = mock.MagicMock()
        model.all_q.return_value = outputs

        val = model.q(input_state, input_action)
        val.forward()

        model.all_q.assert_called_once_with(input_state)

        expected = outputs.d[0, 1].reshape(1, 1)

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)

    def test_all_q(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        input_state = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))

        val = model.all_q(input_state)
        val.forward()

        assert val.shape == (1, n_action)

    def test_argmax_q(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        inputs = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model.all_q = mock.MagicMock()
        model.all_q.return_value = outputs

        val = model.argmax_q(inputs)
        val.forward()

        model.all_q.assert_called_once_with(inputs)

        expected = np.argmax(outputs.d, axis=1)

        assert val.shape == (1, 1)
        assert np.allclose(val.d, expected)

    def test_max_q(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name, n_action=n_action)

        inputs = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model.all_q = mock.MagicMock()
        model.all_q.return_value = outputs

        val = model.max_q(inputs)
        val.forward()

        model.all_q.assert_called_once_with(inputs)

        expected = np.max(outputs.d, axis=1, keepdims=True)

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)
