# Copyright 2022 Sony Group Corporation.
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
import nnabla_rl.distributions as D
import nnabla_rl.environments as E
from nnabla_rl.builders import ModelBuilder
from nnabla_rl.models import QFunction, StochasticPolicy, VFunction
from nnabla_rl.replay_buffer import ReplayBuffer


class RNNPolicyFunction(StochasticPolicy):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _action_dim: int
    _max_action_value: float

    def __init__(self, scope_name: str, action_dim: int):
        super(RNNPolicyFunction, self).__init__(scope_name)
        self._action_dim = action_dim
        self._lstm_state_size = action_dim*2
        self._h = None
        self._c = None

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear1"):
                h = NPF.affine(s, n_outmaps=400)
            h = NF.relu(x=h)
            with nn.parameter_scope("linear2"):
                h = NPF.affine(h, n_outmaps=300)
            h = NF.relu(x=h)
            if not self._is_internal_state_created():
                # automatically create internal states if not provided
                batch_size = h.shape[0]
                self._create_internal_states(batch_size)
            with nn.parameter_scope("linear3"):
                self._h, self._c = NPF.lstm_cell(h, self._h, self._c, self._lstm_state_size)
            h = self._h

            reshaped = NF.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_sigma = NF.split(reshaped, axis=1)
            assert mean.shape == ln_sigma.shape
            assert mean.shape == (s.shape[0], self._action_dim)
            ln_var = ln_sigma * 2.0
        return D.SquashedGaussian(mean=mean, ln_var=ln_var)

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


class RNNQFunction(QFunction):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details

    def __init__(self, scope_name: str):
        super(RNNQFunction, self).__init__(scope_name)
        self._lstm_state_size = 1
        self._h = None
        self._c = None

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            h = NF.concatenate(s, a)
            with nn.parameter_scope("linear1"):
                h = NPF.affine(h, n_outmaps=400)
            h = NF.relu(x=h)
            with nn.parameter_scope("linear2"):
                h = NPF.affine(h, n_outmaps=300)
            h = NF.relu(x=h)
            if not self._is_internal_state_created():
                # automatically create internal states if not provided
                batch_size = h.shape[0]
                self._create_internal_states(batch_size)
            with nn.parameter_scope("linear3"):
                self._h, self._c = NPF.lstm_cell(h, self._h, self._c, self._lstm_state_size)
                h = self._h
        return h

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


class RNNVFunction(VFunction):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details

    def __init__(self, scope_name: str):
        super(RNNVFunction, self).__init__(scope_name)
        self._lstm_state_size = 1
        self._h = None
        self._c = None

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear1"):
                h = NPF.affine(s, n_outmaps=400)
            h = NF.relu(x=h)
            with nn.parameter_scope("linear2"):
                h = NPF.affine(h, n_outmaps=300)
            h = NF.relu(x=h)
            if not self._is_internal_state_created():
                # automatically create internal states if not provided
                batch_size = h.shape[0]
                self._create_internal_states(batch_size)
            with nn.parameter_scope("linear3"):
                self._h, self._c = NPF.lstm_cell(h, self._h, self._c, self._lstm_state_size)
                h = self._h
        return h

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


class TestMMESAC(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuous()
        sac = A.MMESAC(dummy_env)

        assert sac.__name__ == 'MMESAC'

    def test_discrete_action_env_unsupported(self):
        '''
        Check that error occurs when training on discrete action env
        '''

        dummy_env = E.DummyDiscrete()
        with pytest.raises(Exception):
            A.MMESAC(dummy_env)

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''

        dummy_env = E.DummyContinuous()
        sac = A.MMESAC(dummy_env)

        sac.train_online(dummy_env, total_iterations=10)

    def test_run_online_rnn_training(self):
        '''
        Check that no error occurs when calling online training with RNN model
        '''
        class RNNPolicyBuilder(ModelBuilder[StochasticPolicy]):
            def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
                return RNNPolicyFunction(scope_name, action_dim=env_info.action_dim)

        class RNNQFunctionBuilder(ModelBuilder[QFunction]):
            def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
                return RNNQFunction(scope_name)

        class RNNVFunctionBuilder(ModelBuilder[QFunction]):
            def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
                return RNNVFunction(scope_name)

        dummy_env = E.DummyContinuous()
        config = A.MMESACConfig()
        config.num_steps = 2
        config.actor_unroll_steps = 2
        config.actor_burn_in_steps = 2
        config.critic_unroll_steps = 2
        config.critic_burn_in_steps = 2
        config.start_timesteps = 7
        config.batch_size = 2
        sac = A.MMESAC(dummy_env, config=config,
                       policy_builder=RNNPolicyBuilder(),
                       q_function_builder=RNNQFunctionBuilder(),
                       v_function_builder=RNNVFunctionBuilder())

        sac.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''

        batch_size = 5
        dummy_env = E.DummyContinuous()
        config = A.MMESACConfig(batch_size=batch_size)
        sac = A.MMESAC(dummy_env, config=config)

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        sac.train_offline(buffer, total_iterations=10)

    def test_compute_eval_action(self):
        dummy_env = E.DummyContinuous()
        sac = A.MMESAC(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = sac.compute_eval_action(state)

        assert action.shape == dummy_env.action_space.shape

    def test_target_network_initialization(self):
        dummy_env = E.DummyContinuous()
        sac = A.MMESAC(dummy_env)

        # Should be initialized to same parameters
        assert self._has_same_parameters(
            sac._v.get_parameters(), sac._target_v.get_parameters())

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.MMESACConfig(tau=1.1)
        with pytest.raises(ValueError):
            A.MMESACConfig(tau=-0.1)
        with pytest.raises(ValueError):
            A.MMESACConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.MMESACConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.MMESACConfig(start_timesteps=-100)
        with pytest.raises(ValueError):
            A.MMESACConfig(environment_steps=-100)
        with pytest.raises(ValueError):
            A.MMESACConfig(gradient_steps=-100)
        with pytest.raises(ValueError):
            A.MMESACConfig(target_update_interval=-100)

    def _has_same_parameters(self, params1, params2):
        for key in params1.keys():
            if not np.allclose(params1[key].data.data, params2[key].data.data):
                return False
        return True

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuous()
        sac = A.MMESAC(dummy_env)

        sac._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}
        sac._policy_trainer_state = {'pi_loss': 2.}
        sac._v_function_trainer_state = {'v_loss': 1.}

        latest_iteration_state = sac.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'v_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert latest_iteration_state['scalar']['v_loss'] == 1.
        assert latest_iteration_state['scalar']['pi_loss'] == 2.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from tests.testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
