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

import numpy as np
import pytest

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
import nnabla_rl.functions as RF
from nnabla_rl.builders.model_builder import ModelBuilder
from nnabla_rl.models.q_function import ContinuousQFunction, QFunction
from nnabla_rl.replay_buffer import ReplayBuffer


class DummyQFunction(ContinuousQFunction):
    def __init__(self, action_high: np.ndarray, action_low: np.ndarray):
        super(DummyQFunction, self).__init__('dummy')
        self._random_sample_size = 16
        self._action_high = action_high
        self._action_low = action_low

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        batch_size = s.shape[0]

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope('state_conv1'):
                h = NF.relu(NPF.convolution(s, 32, (3, 3), stride=(2, 2)))

            with nn.parameter_scope('state_conv2'):
                h = NF.relu(NPF.convolution(h, 32, (3, 3), stride=(2, 2)))

            with nn.parameter_scope('state_conv3'):
                encoded_state = NF.relu(NPF.convolution(h, 32, (3, 3), stride=(2, 2)))

            with nn.parameter_scope('action_affine1'):
                encoded_action = NF.relu(NPF.affine(a, 32))
                encoded_action = NF.reshape(encoded_action, (batch_size, 32, 1, 1))

            h = encoded_state + encoded_action

            with nn.parameter_scope('affine1'):
                h = NF.relu(NPF.affine(h, 32))

            with nn.parameter_scope('affine2'):
                h = NF.relu(NPF.affine(h, 32))

            with nn.parameter_scope('affine3'):
                q_value = NPF.affine(h, 1)

        return q_value

    def max_q(self, s: nn.Variable) -> nn.Variable:
        return self.q(s, self.argmax_q(s))

    def argmax_q(self, s: nn.Variable) -> nn.Variable:
        tile_size = self._random_sample_size
        tiled_s = self._tile_state(s, tile_size)
        batch_size = s.shape[0]

        def objective_function(a):
            batch_size, sample_size, action_dim = a.shape
            a = a.reshape((batch_size*sample_size, action_dim))
            q_value = self.q(tiled_s, a)
            q_value = q_value.reshape((batch_size, sample_size, 1))
            return q_value

        upper_bound = np.tile(self._action_high, (batch_size, 1))
        lower_bound = np.tile(self._action_low, (batch_size, 1))
        optimized_action = RF.random_shooting_method(
            objective_function,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            sample_size=self._random_sample_size
        )

        return optimized_action

    def _tile_state(self, s, tile_size):
        tile_reps = [tile_size, ] + [1, ] * len(s.shape)
        s = NF.tile(s, tile_reps)
        transpose_reps = [1, 0, ] + list(range(len(s.shape)))[2:]
        s = NF.transpose(s, transpose_reps)
        s = NF.reshape(s, (-1, *s.shape[2:]))
        return s


class DummyQFunctionBuilder(ModelBuilder[QFunction]):
    def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
        return DummyQFunction(action_high=env_info.action_high, action_low=env_info.action_low)


class TestICRA2018QtOpt(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyContinuousImg(image_shape=(3, 64, 64))
        qtopt = A.ICRA2018QtOpt(dummy_env, q_func_builder=DummyQFunctionBuilder())

        assert qtopt.__name__ == 'ICRA2018QtOpt'

    def test_discrete_action_env_unsupported(self):
        '''
        Check that error occurs when training on discrete action env
        '''

        dummy_env = E.DummyDiscreteImg()
        config = A.ICRA2018QtOptConfig()
        with pytest.raises(Exception):
            A.ICRA2018QtOpt(dummy_env, config=config, q_func_builder=DummyQFunctionBuilder())

    def test_run_online_training(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyContinuousImg()
        config = A.ICRA2018QtOptConfig()
        config.start_timesteps = 5
        config.batch_size = 5
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        qtopt = A.ICRA2018QtOpt(dummy_env, config=config, q_func_builder=DummyQFunctionBuilder())

        qtopt.train_online(dummy_env, total_iterations=10)

    def test_run_online_training_multistep(self):
        '''
        Check that no error occurs when calling online training
        '''
        dummy_env = E.DummyContinuousImg()
        config = A.ICRA2018QtOptConfig()
        config.num_steps = 2
        config.start_timesteps = 5
        config.batch_size = 2
        config.learner_update_frequency = 1
        config.target_update_frequency = 1
        qtopt = A.ICRA2018QtOpt(dummy_env, config=config, q_func_builder=DummyQFunctionBuilder())

        qtopt.train_online(dummy_env, total_iterations=10)

    def test_run_offline_training(self):
        '''
        Check that no error occurs when calling offline training
        '''
        dummy_env = E.DummyContinuousImg()
        batch_size = 5
        config = A.ICRA2018QtOptConfig()
        config.batch_size = batch_size
        config.learner_update_frequency = 1
        config.target_update_frequency = 1

        qtopt = A.ICRA2018QtOpt(dummy_env, config=config, q_func_builder=DummyQFunctionBuilder())

        experiences = generate_dummy_experiences(dummy_env, batch_size)
        buffer = ReplayBuffer()
        buffer.append_all(experiences)
        qtopt.train_offline(buffer, total_iterations=5)

    def test_compute_eval_action(self):
        dummy_env = E.DummyDiscreteImg()
        dqn = A.DQN(dummy_env)

        state = dummy_env.reset()
        state = np.float32(state)
        action = dqn.compute_eval_action(state)

        assert action.shape == (1,)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(learning_rate=-0.1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(q_loss_scalar=-0.1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(learner_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(target_update_frequency=-1000)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(start_timesteps=-1000)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(replay_buffer_size=-1000)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(initial_epsilon=1.5)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(final_epsilon=1.1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(test_epsilon=-1000)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(max_explore_steps=-100)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(cem_num_elites=-1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(cem_sample_size=-1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(cem_num_iterations=-1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(cem_alpha=-0.1)
        with pytest.raises(ValueError):
            A.ICRA2018QtOptConfig(random_sample_size=-1)

    def test_latest_iteration_state(self):
        '''
        Check that latest iteration state has the keys and values we expected
        '''

        dummy_env = E.DummyContinuousImg()
        dqn = A.ICRA2018QtOpt(dummy_env, q_func_builder=DummyQFunctionBuilder())

        dqn._q_function_trainer_state = {'q_loss': 0., 'td_errors': np.array([0., 1.])}

        latest_iteration_state = dqn.latest_iteration_state
        assert 'q_loss' in latest_iteration_state['scalar']
        assert 'td_errors' in latest_iteration_state['histogram']
        assert latest_iteration_state['scalar']['q_loss'] == 0.
        assert np.allclose(latest_iteration_state['histogram']['td_errors'], np.array([0., 1.]))


if __name__ == "__main__":
    from testing_utils import generate_dummy_experiences
    pytest.main()
else:
    from ..testing_utils import generate_dummy_experiences
