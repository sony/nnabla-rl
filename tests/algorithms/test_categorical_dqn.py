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

from typing import Dict, Optional, Tuple

import numpy as np
import pytest

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.builders.model_builder import ModelBuilder
from nnabla_rl.models import DiscreteValueDistributionFunction, ValueDistributionFunction
from nnabla_rl.replay_buffer import ReplayBuffer


class RNNValueDistributionFunction(DiscreteValueDistributionFunction):
    def __init__(self, scope_name: str, n_action: int, n_atom: int, v_min: float, v_max: float):
        super().__init__(scope_name, n_action, n_atom, v_min, v_max)
        self._h = None
        self._c = None

        self._lstm_state_size = 512

    def all_probs(self, s: nn.Variable) -> nn.Variable:
        batch_size = s.shape[0]
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = NPF.convolution(s, outmaps=32, stride=(4, 4), kernel=(8, 8))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = NPF.convolution(h, outmaps=64, kernel=(4, 4), stride=(2, 2))
            h = NF.relu(x=h)
            with nn.parameter_scope("conv3"):
                h = NPF.convolution(h, outmaps=64, kernel=(3, 3), stride=(1, 1))
            h = NF.relu(x=h)
            h = NF.reshape(h, shape=(batch_size, -1))
            with nn.parameter_scope("affine1"):
                if not self._is_internal_state_created():
                    # automatically create internal states if not provided
                    batch_size = h.shape[0]
                    self._create_internal_states(batch_size)
                self._h, self._c = NPF.lstm_cell(h, self._h, self._c, self._lstm_state_size)
                h = self._h
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                h = NPF.affine(
                    h, n_outmaps=self._n_action * self._n_atom)
            h = NF.reshape(h, (-1, self._n_action, self._n_atom))
        assert h.shape == (batch_size, self._n_action, self._n_atom)
        return NF.softmax(h, axis=2)

    def is_recurrent(self) -> bool:
        return True

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes: Dict[str, nn.Variable] = {}
        shapes['lstm_hidden'] = (self._lstm_state_size, )
        shapes['lstm_cell'] = (self._lstm_state_size, )
        return shapes

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        states: Dict[str, nn.Variable] = {}
        states['lstm_hidden'] = self._h
        states['lstm_cell'] = self._c
        return states

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        if states is None:
            if self._h is not None:
                self._h.data.zero()
            if self._c is not None:
                self._c.data.zero()
        else:
            self._h = states['lstm_hidden']
            self._c = states['lstm_cell']

    def _create_internal_states(self, batch_size):
        self._h = nn.Variable((batch_size, self._lstm_state_size))
        self._c = nn.Variable((batch_size, self._lstm_state_size))

        self._h.data.zero()
        self._c.data.zero()

    def _is_internal_state_created(self) -> bool:
        return self._h is not None and self._c is not None


class TestCategoricalDQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        categorical_dqn = A.CategoricalDQN(dummy_env)

        assert categorical_dqn.__name__ == 'CategoricalDQN'

    def test_continuous_action_env_unsupported(self):
        '''
        Check that error occurs when training on continuous action env
        '''

        dummy_env = E.DummyContinuous()
        config = A.CategoricalDQNConfig()
        with pytest.raises(Exception):
            A.CategoricalDQN(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.CategoricalDQNConfig()
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        categorical_dqn = A.CategoricalDQN(dummy_env, config=config)

        categorical_dqn.train_online(dummy_env, total_iterations=10)

    def test_run_online_training_multistep(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.CategoricalDQNConfig()
        config.num_steps = 2
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        categorical_dqn = A.CategoricalDQN(dummy_env, config=config)

        categorical_dqn.train_online(dummy_env, total_iterations=10)

    def test_run_online_rnn_training(self):
        '''
        Check that no error occurs when calling online training with RNN model
        '''
        class RNNModelBuilder(ModelBuilder[ValueDistributionFunction]):
            def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
                n_action = env_info.action_dim
                n_atom = algorithm_config.num_atoms
                v_min = algorithm_config.v_min
                v_max = algorithm_config.v_max
                return RNNValueDistributionFunction(scope_name, n_action, n_atom, v_min, v_max)
        dummy_env = E.DummyDiscreteImg()
        config = A.CategoricalDQNConfig()
        config.num_steps = 2
        config.unroll_steps = 2
        config.burn_in_steps = 2
        config.start_timesteps = 7
        config.batch_size = 2
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        categorical_dqn = A.CategoricalDQN(dummy_env, config=config, value_distribution_builder=RNNModelBuilder())

        categorical_dqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        batch_size = 5
        dummy_env = E.DummyDiscreteImg()
        config = A.CategoricalDQNConfig(batch_size=batch_size)
        categorical_dqn = A.CategoricalDQN(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        categorical_dqn.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        categorical_dqn = A.CategoricalDQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = categorical_dqn.compute_eval_action(state)

        assert action.shape == (1, )

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        categorical_dqn = A.CategoricalDQN(dummy_env)

        categorical_dqn._model_trainer_state = {'cross_entropy_loss': 0., 'td_errors': np.array([0., 1.])}

        latest_iteration_state = categorical_dqn.latest_iteration_state
        assert 'cross_entropy_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['cross_entropy_loss'] == 0.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
