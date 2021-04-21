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

from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

import nnabla as nn
import nnabla_rl.model_trainers as MT
from nnabla_rl.environments.dummy import DummyDiscreteImg
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import DQNQFunction


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


class MultiStepTrainerForTest(MT.q_value_trainers.multi_step_trainer.MultiStepTrainer):
    def _update_model(self, models, solvers, batch, training_variables, **kwargs):
        pass

    def _build_training_graph(self, models, training_variables):
        pass

    def _setup_training_variables(self, batch_size):
        pass


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

        n_step_batch = trainer._setup_batch(batch)

        assert np.allclose(batch.s_current, n_step_batch.s_current)
        assert np.allclose(batch.a_current, n_step_batch.a_current)
        assert not np.allclose(batch.reward, n_step_batch.reward)
        assert not np.allclose(batch.gamma, n_step_batch.gamma)
        assert not np.allclose(batch.non_terminal, n_step_batch.non_terminal)

        last_batch = batch[len(batch) - 1]
        assert np.allclose(last_batch.s_next, n_step_batch.s_next)
        assert np.allclose(batch.weight, n_step_batch.weight)


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
