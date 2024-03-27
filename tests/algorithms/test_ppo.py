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

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.algorithms.ppo import _copy_np_array_to_mp_array
from nnabla_rl.builders import ModelBuilder
from nnabla_rl.distributions import Gaussian
from nnabla_rl.models import StochasticPolicy, VFunction
from nnabla_rl.utils.multiprocess import mp_array_from_np_array, mp_to_np_array


class TupleStateActor(StochasticPolicy):
    _action_dim: int

    def __init__(self, scope_name: str, action_dim: int):
        super(TupleStateActor, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable):
        s, *_ = s
        return Gaussian(mean=nn.Variable.from_numpy_array(np.zeros(shape=(s.shape[0], self._action_dim))),
                        ln_var=nn.Variable.from_numpy_array(np.zeros(shape=(s.shape[0], self._action_dim))))


class TupleStateActorBuilder(ModelBuilder[VFunction]):
    def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
        return TupleStateActor(scope_name, env_info.action_dim)


class TupleStateVFunction(VFunction):
    def __init__(self, scope_name: str):
        super(TupleStateVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable):
        s, *_ = s
        v = nn.Variable.from_numpy_array(np.ones(shape=(s.shape[0], 1)))
        return NF.relu(v)


class TupleStateVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self, scope_name: str, env_info, algorithm_config, **kwargs):
        return TupleStateVFunction(scope_name)


class TestPPO(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        assert ppo.__name__ == 'PPO'

    def test_run_online_discrete_env_training(self):
        """Check that no error occurs when calling online training (discrete
        env)"""

        dummy_env = E.DummyDiscreteImg()
        actor_timesteps = 10
        actor_num = 2
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_continuous_env_training(self):
        """Check that no error occurs when calling online training (continuous
        env)"""

        dummy_env = E.DummyContinuous()
        actor_timesteps = 10
        actor_num = 2
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps * actor_num)

    def test_run_online_tuple_state_env_training(self):
        """Check that no error occurs when calling online training (tuple state
        env)"""

        dummy_env = E.DummyTupleStateContinuous()
        actor_timesteps = 10
        actor_num = 2
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num, preprocess_state=False)
        ppo = A.PPO(dummy_env, config=config,
                    v_function_builder=TupleStateVFunctionBuilder(),
                    policy_builder=TupleStateActorBuilder())

        ppo.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_discrete_single_actor(self):
        """Check that no error occurs when calling online training (discrete
        env)"""

        dummy_env = E.DummyDiscreteImg()
        actor_timesteps = 10
        actor_num = 1
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_continuous_single_actor(self):
        """Check that no error occurs when calling online training (continuous
        env)"""

        dummy_env = E.DummyContinuous()
        actor_timesteps = 10
        actor_num = 1
        config = A.PPOConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        ppo = A.PPO(dummy_env, config=config)

        ppo.train_online(dummy_env, total_iterations=actor_timesteps * actor_num)

    def test_run_offline_training(self):
        """Check that no error occurs when calling offline training."""
        dummy_env = E.DummyDiscreteImg()
        ppo = A.PPO(dummy_env)

        with pytest.raises(ValueError):
            ppo.train_offline([], total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.PPOConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.PPOConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.PPOConfig(actor_num=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(actor_timesteps=-1)
        with pytest.raises(ValueError):
            A.PPOConfig(total_timesteps=-1)

    def test_latest_iteration_state(self):
        """Check that latest iteration state has the keys and values we
        expected."""

        dummy_env = E.DummyContinuous()
        ppo = A.PPO(dummy_env)

        ppo._policy_trainer_state = {'pi_loss': 0.}
        ppo._v_function_trainer_state = {'v_loss': 1.}

        latest_iteration_state = ppo.latest_iteration_state
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'v_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['pi_loss'] == 0.
        assert latest_iteration_state['scalar']['v_loss'] == 1.

    def test_copy_np_array_to_mp_array(self):
        shape = (10, 9, 8, 7)
        mp_array_shape_type = (mp_array_from_np_array(np.random.uniform(size=shape)), shape, np.float64)

        test_array = np.random.uniform(size=shape)
        before_copying = mp_to_np_array(mp_array_shape_type[0], shape, dtype=mp_array_shape_type[2])
        assert not np.allclose(before_copying, test_array)

        _copy_np_array_to_mp_array(test_array, mp_array_shape_type)

        after_copying = mp_to_np_array(mp_array_shape_type[0], shape, dtype=mp_array_shape_type[2])
        assert np.allclose(after_copying, test_array)

    def test_copy_tuple_np_array_to_tuple_mp_array_shape_type(self):
        shape = ((10, 9, 8, 7), (6, 5, 4, 3))
        tuple_mp_array_shape_type = tuple(
            [(mp_array_from_np_array(np.random.uniform(size=s)), shape, np.float64) for s in shape]
        )
        tuple_test_array = tuple([np.random.uniform(size=s) for s in shape])

        for mp_ary_shape_type, s, test_ary in zip(tuple_mp_array_shape_type, shape, tuple_test_array):
            before_copying = mp_to_np_array(mp_ary_shape_type[0], s, dtype=mp_ary_shape_type[2])
            assert not np.allclose(before_copying, test_ary)

        _copy_np_array_to_mp_array(tuple_test_array, tuple_mp_array_shape_type)

        for mp_ary_shape_type, s, test_ary in zip(tuple_mp_array_shape_type, shape, tuple_test_array):
            after_copying = mp_to_np_array(mp_ary_shape_type[0], s, dtype=mp_ary_shape_type[2])
            assert np.allclose(after_copying, test_ary)

    def test_copy_np_array_to_tuple_mp_array_shape_type(self):
        shape = ((10, 9, 8, 7), (6, 5, 4, 3))
        tuple_mp_array_shape_type = tuple(
            [(mp_array_from_np_array(np.random.uniform(size=s)), shape, np.float64) for s in shape]
        )
        test_array = np.random.uniform(size=shape[0])

        with pytest.raises(ValueError):
            _copy_np_array_to_mp_array(test_array, tuple_mp_array_shape_type)

    def test_copy_tuple_np_array_to_mp_array_shape_type(self):
        shape = ((10, 9, 8, 7), (6, 5, 4, 3))
        mp_array_shape_type = (mp_array_from_np_array(np.random.uniform(size=shape[0])), shape, np.float64)
        tuple_test_array = tuple([np.random.uniform(size=s) for s in shape])

        with pytest.raises(ValueError):
            _copy_np_array_to_mp_array(tuple_test_array, mp_array_shape_type)


if __name__ == "__main__":
    pytest.main()
