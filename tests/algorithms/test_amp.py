# Copyright 2024 Sony Group Corporation.
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

from unittest.mock import MagicMock

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.algorithms.amp import (_compute_v_target_and_advantage_with_clipping_and_overwriting, _concatenate_state,
                                      _copy_np_array_to_mp_array, _EquallySampleBufferIterator,
                                      _sample_experiences_from_buffers)
from nnabla_rl.environments.amp_env import TaskResult
from nnabla_rl.environments.wrappers.common import FlattenNestedTupleStateWrapper
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.models import VFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.multiprocess import mp_array_from_np_array, mp_to_np_array


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


class TestAMP(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_algorithm_name(self):
        dummy_env = E.DummyAMPEnv()
        amp = A.AMP(dummy_env)

        assert amp.__name__ == 'AMP'

    def test_run_online_amp_env_training(self):
        """Check that no error occurs when calling online training (amp env)"""

        dummy_env = E.DummyAMPEnv()
        actor_timesteps = 10
        actor_num = 2
        config = A.AMPConfig(batch_size=5, actor_timesteps=actor_timesteps, actor_num=actor_num)
        amp = A.AMP(dummy_env, config=config)

        amp.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_amp_goal_env_training(self):
        """Check that no error occurs when calling online training (emp goal
        env)"""

        dummy_env = E.DummyAMPGoalEnv()
        dummy_env = GoalConditionedTupleObservationEnv(dummy_env)
        dummy_env = FlattenNestedTupleStateWrapper(dummy_env)
        actor_timesteps = 10
        actor_num = 2
        config = A.AMPConfig(batch_size=5, actor_timesteps=actor_timesteps,
                             actor_num=actor_num, use_reward_from_env=True)
        amp = A.AMP(dummy_env, config=config)

        amp.train_online(dummy_env, total_iterations=actor_timesteps*actor_num)

    def test_run_online_with_invalid_env_trainig(self):
        """Check that error occurs when calling online training (invalid env,
        not AMPEnv or AMPGoalEnv)"""

        dummy_env = E.DummyContinuous()
        config = A.AMPConfig(batch_size=5)
        amp = A.AMP(dummy_env, config=config)

        with pytest.raises(ValueError):
            amp.train_online(dummy_env)

    def test_run_offline_training(self):
        """Check that no error occurs when calling offline training."""

        dummy_env = E.DummyAMPEnv()
        amp = A.AMP(dummy_env)

        with pytest.raises(ValueError):
            amp.train_offline([], total_iterations=10)

    def test_parameter_range(self):
        with pytest.raises(ValueError):
            A.AMPConfig(gamma=1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            A.AMPConfig(lmb=1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(lmb=-0.1)

        with pytest.raises(ValueError):
            A.AMPConfig(policy_learning_rate=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(policy_momentum=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(policy_weight_decay=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(action_bound_loss_coefficient=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(epsilon=-1.1)

        with pytest.raises(ValueError):
            A.AMPConfig(v_function_learning_rate=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(v_function_momentum=-1.1)

        with pytest.raises(ValueError):
            A.AMPConfig(normalized_advantage_clip=(1.2, -1.1))
        with pytest.raises(ValueError):
            A.AMPConfig(target_value_clip=(1.2, -1.1))

        with pytest.raises(ValueError):
            A.AMPConfig(epochs=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(actor_num=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(batch_size=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(actor_timesteps=-1)

        with pytest.raises(ValueError):
            A.AMPConfig(max_explore_steps=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(final_explore_rate=-0.1)
        with pytest.raises(ValueError):
            A.AMPConfig(final_explore_rate=1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(num_processor_samples=-1)

        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_learning_rate=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_momentum=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_weight_decay=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(action_bound_loss_coefficient=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_extra_regularization_coefficient=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_gradient_penelty_coefficient=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_batch_size=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_epochs=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_reward_scale=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(discriminator_agent_replay_buffer_size=-1)
        with pytest.raises(ValueError):
            A.AMPConfig(lerp_reward_coefficient=-1.1)
        with pytest.raises(ValueError):
            A.AMPConfig(lerp_reward_coefficient=1.1)

    def test_latest_iteration_state(self):
        """Check that latest iteration state has the keys and values we
        expected."""

        dummy_env = E.DummyAMPEnv()
        amp = A.AMP(dummy_env)

        amp._policy_trainer_state = {'pi_loss': 0.}
        amp._v_function_trainer_state = {'v_loss': 1.}
        amp._discriminator_trainer_state = {'reward_loss': 2.}

        latest_iteration_state = amp.latest_iteration_state
        assert 'pi_loss' in latest_iteration_state['scalar']
        assert 'v_loss' in latest_iteration_state['scalar']
        assert 'reward_loss' in latest_iteration_state['scalar']
        assert latest_iteration_state['scalar']['pi_loss'] == 0.
        assert latest_iteration_state['scalar']['v_loss'] == 1.
        assert latest_iteration_state['scalar']['reward_loss'] == 2.

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

    def test_concatenate_state(self):
        s = (np.random.rand(3), np.random.rand(3))
        a = (np.random.rand(4), np.random.rand(4))
        r = (np.random.rand(1), np.random.rand(1))
        non_terminal = (np.random.rand(1), np.random.rand(1))
        n_s = (np.random.rand(3), np.random.rand(3))
        log_prob = (np.random.rand(1), np.random.rand(1))
        non_greedy = (np.random.rand(1), np.random.rand(1))
        e_s = (np.random.rand(3), np.random.rand(3))
        e_a = (np.random.rand(4), np.random.rand(4))
        e_s_next = (np.random.rand(3), np.random.rand(3))
        v_target = (np.random.rand(1), np.random.rand(1))
        advantage = (np.random.rand(1), np.random.rand(1))
        dummy_experiences_per_agent = [[tuple(experience) for experience in
                                       zip(s, a, r, non_terminal, n_s, log_prob,
                                           non_greedy, e_s, e_a, e_s_next, v_target, advantage)]]
        actual_concat_s, actual_concat_e_s = _concatenate_state(dummy_experiences_per_agent)
        assert np.allclose(actual_concat_s, np.stack(s, axis=0))
        assert np.allclose(actual_concat_e_s, np.stack(e_s, axis=0))

    def test_sample_experiences_from_buffers(self):
        buffers = [ReplayBuffer() for _ in range(2)]
        for buffer in buffers:
            buffer.sample = MagicMock(return_value=(((1, ), (2, ), (3, )), {}))

        _sample_experiences_from_buffers(buffers=buffers, batch_size=6)

        # Check all buffer called ones with num samples 3
        for buffer in buffers:
            buffer.sample.assert_called_once_with(num_samples=3)

    @pytest.mark.parametrize("gamma, lmb, value_at_task_fail, value_at_task_success,"
                             "value_clip, expected_adv, expected_vtarg",
                             [[1., 0., 0., 1., None, np.array([[1.], [1.], [1.]]), np.array([[3.], [3.], [3.]])],
                              [1., 1., 0., 1., None, np.array([[3.], [2.], [1.]]), np.array([[5.], [4.], [3.]])],
                              [0.9, 0.7, 0., 1., None, np.array([[1.62152], [1.304], [0.8]]),
                               np.array([[3.62152], [3.304], [2.8]])],
                              [1., 1., 0., 1., (-1.2, 1.2), np.array([[3.], [2.], [1.]]),
                               np.array([[4.2], [3.2], [2.2]])]
                              ])
    def test_compute_v_target_and_advantage_with_clipping_and_overwriting_unknown_task_result(
            self, gamma, lmb, value_at_task_fail, value_at_task_success,
            value_clip, expected_adv, expected_vtarg):
        dummy_v_function = DummyVFunction()
        dummy_experience = self._collect_dummy_experience_unknown_task_result()
        r = np.ones(3)

        actual_vtarg, actual_adv = _compute_v_target_and_advantage_with_clipping_and_overwriting(
            dummy_v_function, dummy_experience, r, gamma, lmb, value_at_task_fail, value_at_task_success, value_clip)

        assert np.allclose(actual_adv, expected_adv)
        assert np.allclose(actual_vtarg, expected_vtarg)

    @pytest.mark.parametrize("gamma, lmb, value_at_task_fail, value_at_task_success,"
                             "value_clip, expected_adv, expected_vtarg",
                             [[1., 0., -1., 1., None, np.array([[1.], [1.], [-2.]]), np.array([[3.], [3.], [0.]])],
                              [1., 1., -1., 1., None, np.array([[0.], [-1.], [-2.]]), np.array([[2.], [1.], [0.]])],
                              [1., 1., -1., 1., (-1.2, 1.2), np.array([[0.8], [-0.2], [-1.2]]),
                               np.array([[2.], [1.], [0.]])]
                              ])
    def test_compute_v_target_and_advantage_with_clipping_and_overwriting_unknown_task_fail(
            self, gamma, lmb, value_at_task_fail, value_at_task_success,
            value_clip, expected_adv, expected_vtarg):
        dummy_v_function = DummyVFunction()
        dummy_experience = self._collect_dummy_experience_unknown_task_result(
            task_result=TaskResult(TaskResult.FAIL.value))
        r = np.ones(3)

        actual_vtarg, actual_adv = _compute_v_target_and_advantage_with_clipping_and_overwriting(
            dummy_v_function, dummy_experience, r, gamma, lmb, value_at_task_fail, value_at_task_success, value_clip)

        assert np.allclose(actual_adv, expected_adv, atol=1e-6)
        assert np.allclose(actual_vtarg, expected_vtarg, atol=1e-6)

    @pytest.mark.parametrize("gamma, lmb, value_at_task_fail, value_at_task_success,"
                             "value_clip, expected_adv, expected_vtarg",
                             [[1., 0., -1., 5., None, np.array([[1.], [1.], [4.]]), np.array([[3.], [3.], [6.]])],
                              [1., 1., -1., 5., None, np.array([[6.], [5.], [4.]]), np.array([[8.], [7.], [6.]])],
                              [1., 1., -1., 5., (-1.2, 1.2), np.array([[6.8], [5.8], [4.8]]),
                               np.array([[8.], [7.], [6.]])]
                              ])
    def test_compute_v_target_and_advantage_with_clipping_and_overwriting_unknown_task_success(
            self, gamma, lmb, value_at_task_fail, value_at_task_success,
            value_clip, expected_adv, expected_vtarg):
        dummy_v_function = DummyVFunction()
        dummy_experience = self._collect_dummy_experience_unknown_task_result(
            task_result=TaskResult(TaskResult.SUCCESS.value))
        r = np.ones(3)

        actual_vtarg, actual_adv = _compute_v_target_and_advantage_with_clipping_and_overwriting(
            dummy_v_function, dummy_experience, r, gamma, lmb, value_at_task_fail, value_at_task_success, value_clip)

        assert np.allclose(actual_adv, expected_adv, atol=1e-6)
        assert np.allclose(actual_vtarg, expected_vtarg, atol=1e-6)

    def _collect_dummy_experience_unknown_task_result(self,
                                                      num_episodes=1, episode_length=3,
                                                      task_result=TaskResult(TaskResult.UNKNOWN.value)):
        experience = []
        for _ in range(num_episodes):
            for i in range(episode_length):
                s_current = np.ones(1, )
                a = np.ones(1, )
                s_next = np.ones(1, )
                r = np.ones(1, )
                non_terminal = np.ones(1, )
                info = {"task_result": TaskResult(0)}

                if i == episode_length-1:
                    non_terminal = np.zeros(1, )
                    info = {"task_result": task_result}

                experience.append((s_current, a, r, non_terminal, s_next, info))
        return experience


class TestEquallySampleBufferIterator():
    def test_equally_sample_buffer_iterator_iterates_correct_number_of_times(self):
        buffer_size = 5
        buffers = [ReplayBuffer(buffer_size) for _ in range(2)]

        for i, buffer in enumerate(buffers):
            buffer.append_all(np.arange(buffer_size) * (i+1))

        batch_size = 6
        total_num_iterations = 10
        buffer_iterator = _EquallySampleBufferIterator(total_num_iterations, buffers, batch_size=batch_size)

        for _ in range(total_num_iterations):
            batch = buffer_iterator.next()
            assert len(batch) == batch_size

        with pytest.raises(StopIteration):
            buffer_iterator.next()

    def test_equally_sample_buffer_iterator_iterates_correct_data(self):
        buffer_size = 4
        buffers = [ReplayBuffer(buffer_size) for _ in range(2)]

        for i, buffer in enumerate(buffers):
            dummy_experience = [((j+1)*(i+1), ) for j in range(buffer_size)]
            buffer.append_all(dummy_experience)

        batch_size = 4
        total_num_iterations = 3
        buffer_iterator = _EquallySampleBufferIterator(total_num_iterations, buffers, batch_size=batch_size)

        expected = [[(1,), (2,), (2,), (4,)], [(3,), (4,), (6,), (8,)], [(1,), (2,), (2,), (4,)]]
        for actual_batch, expected_batch in zip(buffer_iterator, expected):
            assert len(actual_batch) == batch_size
            assert tuple(actual_batch) == tuple(expected_batch)


if __name__ == "__main__":
    pytest.main()
