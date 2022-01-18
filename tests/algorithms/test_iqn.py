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

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pytest

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
import nnabla_rl.functions as RF
import nnabla_rl.parametric_functions as RPF
from nnabla_rl.builders.model_builder import ModelBuilder
from nnabla_rl.models import IQNQuantileFunction, StateActionQuantileFunction
from nnabla_rl.replay_buffer import ReplayBuffer


class RNNStateActionQuantileFunction(IQNQuantileFunction):
    def __init__(self,
                 scope_name: str,
                 n_action: int,
                 embedding_dim: int,
                 K: int,
                 risk_measure_function: Callable[[nn.Variable], nn.Variable]):
        super().__init__(scope_name, n_action, embedding_dim, K, risk_measure_function)
        self._h = None
        self._c = None

        self._lstm_state_size = 512

    def all_quantile_values(self, s: nn.Variable, tau: nn.Variable) -> nn.Variable:
        encoded = self._encode(s, tau.shape[-1])
        embedding = self._compute_embedding(tau, encoded.shape[-1])

        assert embedding.shape == encoded.shape

        with nn.parameter_scope(self.scope_name):
            h = encoded * embedding
            with nn.parameter_scope("affine1"):
                if not self._is_internal_state_created():
                    # automatically create internal states if not provided
                    batch_size = h.shape[0]
                    self._create_internal_states(batch_size)
                # This is just a workaround to enable using lstm state
                hidden_state = RF.expand_dims(self._h, axis=1)
                hidden_state = NF.broadcast(hidden_state, shape=(self._h.shape[0], tau.shape[-1], self._h.shape[-1]))
                cell_state = RF.expand_dims(self._c, axis=1)
                cell_state = NF.broadcast(cell_state, shape=(self._c.shape[0], tau.shape[-1], self._c.shape[-1]))
                hidden_state, cell_state = RPF.lstm_cell(
                    h, hidden_state, cell_state, self._lstm_state_size, base_axis=2)
                h = hidden_state
                # Save only the state of first sample for the next timestep
                self._h, *_ = NF.split(hidden_state, axis=1)
                self._c, *_ = NF.split(cell_state, axis=1)
            h = NF.relu(x=h)
            with nn.parameter_scope("affine2"):
                return_samples = NPF.affine(h, n_outmaps=self._n_action, base_axis=2)
        assert return_samples.shape == (s.shape[0], tau.shape[-1], self._n_action)
        return return_samples

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


class TestIQN(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        assert iqn.__name__ == 'IQN'

    def test_continuous_action_env_unsupported(self):
        '''
        Check that error occurs when training on continuous action env
        '''

        dummy_env = E.DummyContinuous()
        config = A.IQNConfig()
        with pytest.raises(Exception):
            A.IQN(dummy_env, config=config)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.IQNConfig()
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        iqn = A.IQN(dummy_env, config=config)

        iqn.train_online(dummy_env, total_iterations=5)

    def test_run_online_training_multistep(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.IQNConfig()
        config.num_steps = 2
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        iqn = A.IQN(dummy_env, config=config)

        iqn.train_online(dummy_env, total_iterations=5)

    def test_run_online_rnn_training(self):
        '''
        Check that no error occurs when calling online training with RNN model
        '''
        class RNNModelBuilder(ModelBuilder[StateActionQuantileFunction]):
            def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
                risk_measure_function = kwargs['risk_measure_function']
                return RNNStateActionQuantileFunction(scope_name,
                                                      env_info.action_dim,
                                                      algorithm_config.embedding_dim,
                                                      K=algorithm_config.K,
                                                      risk_measure_function=risk_measure_function)
        dummy_env = E.DummyDiscreteImg()
        config = A.IQNConfig()
        config.num_steps = 2
        config.unroll_steps = 2
        config.burn_in_steps = 2
        config.start_timesteps = 7
        config.batch_size = 2
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        qrdqn = A.IQN(dummy_env, config=config, quantile_function_builder=RNNModelBuilder())

        qrdqn.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        dummy_env = E.DummyDiscreteImg()
        batch_size = 5
        config = A.IQNConfig()
        config.batch_size = batch_size
        config.learner_update_frequency = 1
        config.target_update_frequency = 1

        iqn = A.IQN(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        iqn.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = iqn.compute_eval_action(state)

        assert action.shape == (1, )

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.IQNConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.IQNConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.IQNConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(learner_update_frequency=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(max_explore_steps=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(learning_rate=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(initial_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(final_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(test_epsilon=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(K=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(N=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(N_prime=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(kappa=-1)
        with pytest.raises(ValueError):
            A.IQNConfig(embedding_dim=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyDiscreteImg()
        iqn = A.IQN(dummy_env)

        iqn._quantile_function_trainer_state = {'q_loss': 0.}

        latest_iteration_state = iqn.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['q_loss'] == 0.


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
