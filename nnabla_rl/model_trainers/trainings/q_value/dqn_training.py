# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import nnabla.functions as NF

from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import \
    QFunction, ValueDistributionFunction, QuantileDistributionFunction, StateActionQuantileFunction, Model


class _QFunctionDQNTraining(Training):
    _target_function: QFunction

    def __init__(self, target_function: QFunction):
        self._target_function = target_function

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        max_q_value = self._target_function.max_q(s_next)
        return reward + gamma * non_terminal * max_q_value


class _ValueDistributionFunctionDQNTraining(Training):
    _target_function: ValueDistributionFunction

    def __init__(self, target_function: ValueDistributionFunction):
        self._target_function = target_function

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        N = self._target_function._n_atom
        v_max = self._target_function._v_max
        v_min = self._target_function._v_min

        target_atom_probabilities = self._target_function.probabilities(s_next)
        a_star = self._target_function.argmax_q_from_probabilities(target_atom_probabilities)
        pj = self._target_function._probabilities_of(target_atom_probabilities, a_star)

        delta_z = (v_max - v_min) / (N - 1)
        z = np.asarray([v_min + i * delta_z for i in range(N)])
        z = np.broadcast_to(array=z, shape=(batch_size, N))
        z = nn.Variable.from_numpy_array(z)
        target = reward + non_terminal * gamma * z
        Tz = NF.clip_by_value(target, v_min, v_max)
        assert Tz.shape == (batch_size, N)

        mi = self._compute_projection(Tz, pj, N, v_max, v_min)
        return mi

    def _compute_projection(self, Tz, pj, N, v_max, v_min):
        batch_size = Tz.shape[0]
        delta_z = (v_max - v_min) / (N - 1)

        bj = (Tz - v_min) / delta_z
        bj = NF.clip_by_value(bj, 0, N - 1)

        lower = NF.floor(bj)
        upper = NF.ceil(bj)

        ml_indices = lower
        mu_indices = upper

        mi = nn.Variable.from_numpy_array(np.zeros(shape=(batch_size, N), dtype=np.float32))
        # Fix upper - bj = bj - lower = 0 (Prevent not getting both 0. upper - l must always be 1)
        # upper - bj = (1 + lower) - bj
        upper = 1 + lower

        result_upper = NF.scatter_add(mi, ml_indices, pj * (upper - bj), axis=-1)
        result_lower = NF.scatter_add(mi, mu_indices, pj * (bj - lower), axis=-1)

        return (result_upper + result_lower)


class _QuantileDistributionFunctionDQNTraining(Training):
    _target_function: QuantileDistributionFunction

    def __init__(self, target_function: QuantileDistributionFunction):
        self._target_function = target_function

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        target_quantiles = self._target_function.quantiles(s_next)
        a_star = self._target_function.argmax_q_from_quantiles(target_quantiles)

        theta_j = self._target_function._quantiles_of(target_quantiles, a_star)
        Ttheta_j = reward + non_terminal * gamma * theta_j
        return Ttheta_j


class _StateActionQuantileFunctionDQNTraining(Training):
    _target_function: StateActionQuantileFunction

    def __init__(self, target_function: StateActionQuantileFunction):
        self._target_function = target_function

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        K = kwargs['K']
        N_prime = kwargs['N_prime']

        batch_size = s_next.shape[0]

        tau_k = self._target_function._sample_risk_measured_tau(shape=(batch_size, K))
        return_samples = self._target_function.return_samples(s_next, tau_k)
        a_star = self._target_function.argmax_q_from_return_samples(return_samples)

        tau_j = self._target_function._sample_tau(shape=(batch_size, N_prime))
        target_returns_samples = self._target_function.return_samples(s_next, tau_j)
        Z_tau_j = self._target_function._return_samples_of(target_returns_samples, a_star)
        assert Z_tau_j.shape == (batch_size, N_prime)
        target = reward + non_terminal * gamma * Z_tau_j
        return target


class DQNTraining(Training):
    _delegate: Training

    def __init__(self, train_function: Model, target_function: Model):
        if type(train_function) is not type(target_function):
            raise ValueError
        if isinstance(target_function, ValueDistributionFunction):
            self._delegate = _ValueDistributionFunctionDQNTraining(target_function)
        elif isinstance(target_function, QuantileDistributionFunction):
            self._delegate = _QuantileDistributionFunctionDQNTraining(target_function)
        elif isinstance(target_function, StateActionQuantileFunction):
            self._delegate = _StateActionQuantileFunctionDQNTraining(target_function)
        elif isinstance(target_function, QFunction):
            self._delegate = _QFunctionDQNTraining(target_function)
        else:
            raise NotImplementedError(f'No training implementation for class: {target_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
