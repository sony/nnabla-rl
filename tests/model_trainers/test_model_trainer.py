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
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingVariables, rnn_support
from nnabla_rl.models.model import Model


class EmptyModel(Model):
    def __init__(self, scope_name):
        super(EmptyModel, self).__init__(scope_name)

    def __call__(self, s):
        return s


class EmptyRnnModel(EmptyModel):
    def __init__(self, scope_name: str):
        super().__init__(scope_name)
        self._internal_state_shape = (10, )
        self._fake_internal_state = None

    def is_recurrent(self) -> bool:
        return True

    def internal_state_shapes(self):
        return {'fake': self._internal_state_shape}

    def set_internal_states(self, states):
        self._fake_internal_state = states['fake']

    def get_internal_states(self):
        return {'fake': self._fake_internal_state}

    def __call__(self, s):
        self._fake_internal_state = self._fake_internal_state * 2
        return s


class TestModelTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_assert_no_duplicate_model_without_duplicates(self):
        models = [EmptyModel('model1'), EmptyModel('model2'), EmptyModel('model3')]
        ModelTrainer._assert_no_duplicate_model(models)

    def test_assert_no_duplicate_model_with_duplicates(self):
        duplicate_models = [EmptyModel('model1'), EmptyModel('model2'), EmptyModel('model1'), EmptyModel('model3')]
        with pytest.raises(AssertionError):
            ModelTrainer._assert_no_duplicate_model(duplicate_models)

    def test_rnn_support_with_reset_on_terminal(self):
        scope_name = 'test'
        test_model = EmptyRnnModel(scope_name)
        batch_size = 3
        num_steps = 3

        training_variables = None
        for _ in range(num_steps):
            train_rnn_states = self._create_fake_internal_states(batch_size, test_model)
            non_terminal = self._create_fake_non_terminals(batch_size)
            training_variables = TrainingVariables(batch_size,
                                                   non_terminal=non_terminal,
                                                   rnn_states=train_rnn_states,
                                                   next_step_variables=training_variables)

        prev_rnn_states = {}
        config = TrainerConfig(unroll_steps=num_steps, reset_on_terminal=True)
        for variables in training_variables:
            train_rnn_states = variables.rnn_states
            with rnn_support(test_model, prev_rnn_states, train_rnn_states, variables, config):
                actual_states = test_model.get_internal_states()

                if variables.is_initial_step():
                    expected_states = variables.rnn_states[scope_name]
                    self._assert_have_same_states(actual_states, expected_states)
                else:
                    prev_states = prev_rnn_states[scope_name]
                    train_states = variables.rnn_states[scope_name]
                    prev_non_terminal = variables.prev_step_variables.non_terminal
                    expected_states = {}
                    for key in prev_states.keys():
                        expected_state = prev_states[key] * prev_non_terminal + \
                            (1 - prev_non_terminal) * train_states[key]
                        expected_states[key] = expected_state
                    self._assert_have_same_states(actual_states, expected_states)

                # just call to update internal states
                test_model(None)
            assert not len(prev_rnn_states) == 0
            actual_states = test_model.get_internal_states()
            expected_states = prev_rnn_states[scope_name]
            self._assert_have_same_states(actual_states, expected_states)

    def test_rnn_support_without_reset_on_terminal(self):
        scope_name = 'test'
        test_model = EmptyRnnModel(scope_name)
        batch_size = 3
        num_steps = 3

        training_variables = None
        for _ in range(num_steps):
            train_rnn_states = self._create_fake_internal_states(batch_size, test_model)
            non_terminal = self._create_fake_non_terminals(batch_size)
            training_variables = TrainingVariables(batch_size,
                                                   non_terminal=non_terminal,
                                                   rnn_states=train_rnn_states,
                                                   next_step_variables=training_variables)

        prev_rnn_states = {}
        config = TrainerConfig(unroll_steps=num_steps, reset_on_terminal=False)
        for variables in training_variables:
            train_rnn_states = variables.rnn_states
            with rnn_support(test_model, prev_rnn_states, train_rnn_states, variables, config):
                actual_states = test_model.get_internal_states()

                if variables.is_initial_step():
                    expected_states = variables.rnn_states[scope_name]
                    self._assert_have_same_states(actual_states, expected_states)
                else:
                    prev_states = prev_rnn_states[scope_name]
                    expected_states = {}
                    for key in prev_states.keys():
                        expected_state = prev_states[key]
                        expected_states[key] = expected_state
                    self._assert_have_same_states(actual_states, expected_states)

                # just call to update internal states
                test_model(None)
            assert not len(prev_rnn_states) == 0
            actual_states = test_model.get_internal_states()
            expected_states = prev_rnn_states[scope_name]
            self._assert_have_same_states(actual_states, expected_states)

    def test_rnn_support_with_non_rnn_model(self):
        scope_name = 'test'
        test_model = EmptyModel(scope_name)
        batch_size = 3
        num_steps = 3

        training_variables = None
        for _ in range(num_steps):
            non_terminal = self._create_fake_non_terminals(batch_size)
            training_variables = TrainingVariables(batch_size,
                                                   non_terminal=non_terminal,
                                                   next_step_variables=training_variables)
        train_rnn_states = {}
        prev_rnn_states = {}
        config = TrainerConfig(num_steps)
        for variables in training_variables:
            with rnn_support(test_model, prev_rnn_states, train_rnn_states, variables, config):
                test_model(None)
            assert len(prev_rnn_states) == 0

    def _assert_have_same_states(self, actual_states, expected_states):
        for key in actual_states.keys():
            actual = actual_states[key]
            expected = expected_states[key]
            nn.forward_all([actual, expected])
            np.testing.assert_allclose(actual.d, expected.d)

    def _create_fake_non_terminals(self, batch_size):
        non_terminals = np.float32(np.random.randint(low=0, high=2, size=(batch_size, 1)))
        return nn.Variable.from_numpy_array(non_terminals)

    def _create_fake_internal_states(self, batch_size, model):
        states = {}
        for key, shape in model.internal_state_shapes().items():
            state = np.random.normal(size=(batch_size, *shape))
            states[key] = nn.Variable.from_numpy_array(state)
        internal_states = {model.scope_name: states}
        return internal_states


if __name__ == "__main__":
    pytest.main()
