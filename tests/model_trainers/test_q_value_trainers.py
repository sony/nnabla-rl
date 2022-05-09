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

from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.model_trainers as MT
from nnabla_rl.environments.dummy import DummyDiscreteImg
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import LossIntegration, TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.q_value.dqn_q_trainer import DQNQTrainer
from nnabla_rl.models import DQNQFunction, DRQNQFunction


class TestC51ValueDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


class TestQuantileDistributionFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    def test_precompute_tau_hat(self):
        n_quantiles = 100

        expected = np.empty(shape=(n_quantiles,))
        prev_tau = 0.0

        for i in range(0, n_quantiles):
            tau = (i + 1) / n_quantiles
            expected[i] = (prev_tau + tau) / 2.0
            prev_tau = tau

        actual = MT.q_value.QRDQNQTrainer._precompute_tau_hat(n_quantiles)

        assert np.allclose(expected, actual)


class TestSquaredTDQFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    @pytest.mark.parametrize('num_steps', [1, 2])
    @pytest.mark.parametrize('unroll_steps', [1, 2, 3])
    @pytest.mark.parametrize('burn_in_steps', [0, 1, 2])
    @pytest.mark.parametrize('loss_integration', [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_non_rnn_model(self, num_steps, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        train_q = DQNQFunction('stub', n_action=env_info.action_dim)
        target_q = train_q.deepcopy('stub2')
        # Using DQN Q trainer as representative trainer
        config = MT.q_value_trainers.DQNQTrainerConfig(unroll_steps=unroll_steps,
                                                       burn_in_steps=burn_in_steps,
                                                       num_steps=num_steps,
                                                       loss_integration=loss_integration)
        DQNQTrainer(train_functions=train_q, solvers={}, target_function=target_q, env_info=env_info, config=config)

        # pass: If no ecror occurs

    @pytest.mark.parametrize('num_steps', [1, 2])
    @pytest.mark.parametrize('unroll_steps', [1, 2, 3])
    @pytest.mark.parametrize('burn_in_steps', [0, 1, 2])
    @pytest.mark.parametrize('loss_integration', [LossIntegration.LAST_TIMESTEP_ONLY, LossIntegration.ALL_TIMESTEPS])
    def test_with_rnn_model(self, num_steps, unroll_steps, burn_in_steps, loss_integration):
        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        train_q = DRQNQFunction('stub',  n_action=env_info.action_dim)
        target_q = train_q.deepcopy('stub2')
        # Using DQN Q trainer as representative trainer
        config = MT.q_value_trainers.DQNQTrainerConfig(unroll_steps=unroll_steps,
                                                       burn_in_steps=burn_in_steps,
                                                       num_steps=num_steps,
                                                       loss_integration=loss_integration)
        DQNQTrainer(train_functions=train_q, solvers={}, target_function=target_q, env_info=env_info, config=config)

        # pass: If no ecror occurs


class MultiStepTrainerForTest(MT.q_value_trainers.multi_step_trainer.MultiStepTrainer):
    def _update_model(self, models, solvers, batch, training_variables, **kwargs):
        pass

    def support_rnn(self):
        return True

    def _build_training_graph(self, models, training_variables):
        pass

    def _setup_training_variables(self, batch_size):
        return TrainingVariables(batch_size)

    @property
    def loss_variables(self):
        return {}


class TestMultiStepTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()

    @patch("nnabla_rl.model_trainers.q_value_trainers.multi_step_trainer.MultiStepTrainer.__abstractmethods__", set())
    def test_n_step_setup_batch(self):
        batch_size = 5
        num_steps = 5
        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        train_q = DQNQFunction('stub',  n_action=env_info.action_dim)
        config = MT.q_value_trainers.multi_step_trainer.MultiStepTrainerConfig(num_steps=num_steps)
        trainer = MultiStepTrainerForTest(models=train_q, solvers={}, env_info=env_info, config=config)

        batch = _generate_batch(batch_size, num_steps, env_info)
        assert len(batch) == num_steps

        actual_n_step_batch = trainer._setup_batch(batch)

        assert np.allclose(batch.s_current, actual_n_step_batch.s_current)
        assert np.allclose(batch.a_current, actual_n_step_batch.a_current)
        assert not np.allclose(batch.reward, actual_n_step_batch.reward)
        assert not np.allclose(batch.gamma, actual_n_step_batch.gamma)
        assert not np.allclose(batch.non_terminal, actual_n_step_batch.non_terminal)

        last_batch = batch[len(batch) - 1]
        assert np.allclose(last_batch.s_next, actual_n_step_batch.s_next)
        assert np.allclose(batch.weight, actual_n_step_batch.weight)

        expected_n_step_batch = self._expected_batch(batch, num_steps)
        assert np.allclose(expected_n_step_batch.reward, actual_n_step_batch.reward)
        assert np.allclose(expected_n_step_batch.gamma, actual_n_step_batch.gamma)
        assert np.allclose(expected_n_step_batch.non_terminal, actual_n_step_batch.non_terminal)

    @patch("nnabla_rl.model_trainers.q_value_trainers.multi_step_trainer.MultiStepTrainer.__abstractmethods__", set())
    def test_rnn_n_step_setup_batch(self):
        batch_size = 5
        num_steps = 5
        unroll_steps = 3
        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        train_q = DRQNQFunction('stub',  n_action=env_info.action_dim)
        config = MT.q_value_trainers.multi_step_trainer.MultiStepTrainerConfig(num_steps=num_steps,
                                                                               unroll_steps=unroll_steps)
        trainer = MultiStepTrainerForTest(models=train_q, solvers={}, env_info=env_info, config=config)

        batch = _generate_batch(batch_size, num_steps + unroll_steps - 1, env_info)
        assert len(batch) == num_steps + unroll_steps - 1

        actual_n_step_batch = trainer._setup_batch(batch)
        assert len(actual_n_step_batch) == unroll_steps

        for i, actual_n_step_batch in enumerate(actual_n_step_batch):
            expected_n_step_batch = self._expected_batch(batch[i], num_steps)
            assert np.allclose(expected_n_step_batch.s_current, actual_n_step_batch.s_current)
            assert np.allclose(expected_n_step_batch.a_current, actual_n_step_batch.a_current)
            assert np.allclose(expected_n_step_batch.reward, actual_n_step_batch.reward)
            assert np.allclose(expected_n_step_batch.gamma, actual_n_step_batch.gamma)
            assert np.allclose(expected_n_step_batch.non_terminal, actual_n_step_batch.non_terminal)
            assert np.allclose(expected_n_step_batch.s_next, actual_n_step_batch.s_next)
            assert np.allclose(expected_n_step_batch.weight, actual_n_step_batch.weight)

    @patch("nnabla_rl.model_trainers.q_value_trainers.multi_step_trainer.MultiStepTrainer.__abstractmethods__", set())
    def test_rnn_with_burnin_n_step_setup_batch(self):
        batch_size = 5
        num_steps = 5
        unroll_steps = 3
        burn_in_steps = 2
        env_info = EnvironmentInfo.from_env(DummyDiscreteImg())

        train_q = DRQNQFunction('stub',  n_action=env_info.action_dim)
        config = MT.q_value_trainers.multi_step_trainer.MultiStepTrainerConfig(num_steps=num_steps,
                                                                               unroll_steps=unroll_steps,
                                                                               burn_in_steps=burn_in_steps)
        trainer = MultiStepTrainerForTest(models=train_q, solvers={}, env_info=env_info, config=config)

        batch = _generate_batch(batch_size, num_steps + unroll_steps + burn_in_steps - 1, env_info)
        assert len(batch) == num_steps + unroll_steps + burn_in_steps - 1

        actual_n_step_batch = trainer._setup_batch(batch)
        assert len(actual_n_step_batch) == unroll_steps + burn_in_steps

        for i, actual_n_step_batch in enumerate(actual_n_step_batch):
            expected_n_step_batch = self._expected_batch(batch[i], num_steps)
            assert np.allclose(expected_n_step_batch.s_current, actual_n_step_batch.s_current)
            assert np.allclose(expected_n_step_batch.a_current, actual_n_step_batch.a_current)
            assert np.allclose(expected_n_step_batch.reward, actual_n_step_batch.reward)
            assert np.allclose(expected_n_step_batch.gamma, actual_n_step_batch.gamma)
            assert np.allclose(expected_n_step_batch.non_terminal, actual_n_step_batch.non_terminal)
            assert np.allclose(expected_n_step_batch.s_next, actual_n_step_batch.s_next)
            assert np.allclose(expected_n_step_batch.weight, actual_n_step_batch.weight)

    def _expected_batch(self, training_batch: TrainingBatch, num_steps) -> TrainingBatch:
        if num_steps == 1:
            return training_batch
        else:
            n_step_non_terminal = np.copy(training_batch.non_terminal)
            n_step_reward = np.copy(training_batch.reward)
            n_step_gamma = np.copy(training_batch.gamma)
            n_step_state = np.copy(training_batch.s_next)
            next_batch = training_batch.next_step_batch

            for _ in range(num_steps - 1):
                # Do not add reward if previous state is terminal state
                n_step_reward += next_batch.reward * n_step_non_terminal * n_step_gamma
                n_step_non_terminal *= next_batch.non_terminal
                n_step_gamma *= next_batch.gamma
                n_step_state = next_batch.s_next

                next_batch = next_batch.next_step_batch

            return TrainingBatch(batch_size=training_batch.batch_size,
                                 s_current=training_batch.s_current,
                                 a_current=training_batch.a_current,
                                 reward=n_step_reward,
                                 gamma=n_step_gamma,
                                 non_terminal=n_step_non_terminal,
                                 s_next=n_step_state,
                                 weight=training_batch.weight,
                                 extra=training_batch.extra,
                                 next_step_batch=None)


def _generate_batch(batch_size, num_steps, env_info) -> TrainingBatch:
    state_dim = env_info.state_dim
    action_num = env_info.action_dim

    head_batch: Optional[TrainingBatch] = None
    tail_batch: Optional[TrainingBatch] = None
    s_current = np.random.normal(size=(batch_size, state_dim))
    for _ in range(num_steps):
        a_current = np.random.randint(action_num, size=(batch_size, 1)).astype('float32')
        reward = np.random.normal(size=(batch_size, 1))
        gamma = 0.99
        non_terminal = np.random.randint(2, size=(batch_size, 1)).astype('float32')
        s_next = np.random.normal(size=(batch_size, state_dim))
        weight = np.random.normal(size=(batch_size, 1))

        batch = TrainingBatch(batch_size=batch_size,
                              s_current=s_current,
                              a_current=a_current,
                              reward=reward,
                              gamma=gamma,
                              non_terminal=non_terminal,
                              s_next=s_next,
                              weight=weight)
        if head_batch is None:
            head_batch = batch
        if tail_batch is None:
            tail_batch = head_batch
        else:
            tail_batch.next_step_batch = batch
            tail_batch = batch
        s_current = s_next
    assert head_batch is not None
    return head_batch


if __name__ == "__main__":
    pytest.main()
