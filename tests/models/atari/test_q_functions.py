import pytest
from unittest import mock

import numpy as np

import nnabla as nn

from nnabla_rl.models.atari.q_functions import DQNQFunction


class TestDQNQFunction(object):
    def test_scope_name(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name,
                             state_shape=state_shape,
                             n_action=n_action)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name,
                             state_shape=state_shape,
                             n_action=n_action)

        # Fake input to initialize parameters
        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        input_action = nn.Variable.from_numpy_array(np.ones((1, 1)))
        model.q(input_state, input_action)

        assert len(model.get_parameters()) == 10

    def test_call(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name,
                             state_shape=state_shape,
                             n_action=n_action)

        input_state = nn.Variable.from_numpy_array(
            np.random.rand(1, *state_shape))
        input_action = nn.Variable.from_numpy_array(np.ones((1, 1)))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model._predict_q_values = mock.MagicMock()
        model._predict_q_values.return_value = outputs

        val = model.q(input_state, input_action)
        val.forward()

        model._predict_q_values.assert_called_once_with(input_state)

        expected = outputs.d[0, 1].reshape(1, 1)

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)

    def test_argmax(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name,
                             state_shape=state_shape,
                             n_action=n_action)

        inputs = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model._predict_q_values = mock.MagicMock()
        model._predict_q_values.return_value = outputs

        val = model.argmax(inputs)
        val.forward()

        model._predict_q_values.assert_called_once_with(inputs)

        expected = np.argmax(outputs.d, axis=1)

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)

    def test_maximum(self):
        nn.clear_parameters()

        state_shape = (4, 84, 84)
        n_action = 4
        scope_name = "test"
        model = DQNQFunction(scope_name=scope_name,
                             state_shape=state_shape,
                             n_action=n_action)

        inputs = nn.Variable.from_numpy_array(np.random.rand(1, *state_shape))
        outputs = nn.Variable.from_numpy_array(np.random.rand(1, n_action))

        model._predict_q_values = mock.MagicMock()
        model._predict_q_values.return_value = outputs

        val = model.maximum(inputs)
        val.forward()

        model._predict_q_values.assert_called_once_with(inputs)

        expected = np.max(outputs.d, axis=1, keepdims=True)

        assert val.shape == expected.shape
        assert np.allclose(val.d, expected)
