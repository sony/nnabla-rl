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

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import Training, TrainingVariables
from nnabla_rl.models import Model, QFunction, StateActionQuantileFunction


def _pi(q_values: nn.Variable, max_q: nn.Variable, tau: float):
    return NF.softmax((q_values - max_q) / tau)


def _all_tau_log_pi(q_values: nn.Variable, max_q: nn.Variable, tau: float):
    logsumexp = tau * NF.log(NF.sum(x=NF.exp((q_values - max_q) / tau),
                                    axis=(q_values.ndim - 1), keepdims=True))
    return q_values - max_q - logsumexp


def _tau_log_pi(q_k: nn.Variable, q_values: nn.Variable, max_q: nn.Variable, tau: float):
    logsumexp = tau * NF.log(NF.sum(x=NF.exp((q_values - max_q) / tau),
                                    axis=(q_values.ndim - 1), keepdims=True))
    return q_k - max_q - logsumexp


class _QFunctionMunchausenRLTraining(Training):
    _target_function: QFunction
    _tau: float
    _alpha: float
    _clip_min: float
    _clip_max: float

    def __init__(self, target_function: QFunction, tau: float, alpha: float, clip_min: float, clip_max: float):
        self._target_function = target_function
        self._tau = tau
        self._alpha = alpha
        self._clip_min = clip_min
        self._clip_max = clip_max

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        s_current = training_variables.s_current
        a_current = training_variables.a_current
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        all_next_q = self._target_function.all_q(s_next)
        max_next_q = self._target_function.max_q(s_next)
        pi = _pi(all_next_q, max_next_q, tau=self._tau)
        all_tau_log_pi = _all_tau_log_pi(all_next_q, max_next_q, self._tau)
        assert pi.shape == all_next_q.shape
        assert pi.shape == all_tau_log_pi.shape
        soft_q_target = NF.sum(pi * (all_next_q - all_tau_log_pi), axis=(pi.ndim - 1),  keepdims=True)

        current_q = self._target_function.q(s_current, a_current)
        all_current_q = self._target_function.all_q(s_current)
        max_current_q = self._target_function.max_q(s_current)
        tau_log_pi = _tau_log_pi(current_q, all_current_q, max_current_q, self._tau)
        clipped_tau_log_pi = NF.clip_by_value(tau_log_pi, self._clip_min, self._clip_max)
        return reward + self._alpha * clipped_tau_log_pi + gamma * non_terminal * soft_q_target


class _StateActionQuantileFunctionMunchausenRLTraining(Training):
    _target_function: StateActionQuantileFunction
    _tau: float
    _alpha: float
    _clip_min: float
    _clip_max: float

    def __init__(self, target_function: StateActionQuantileFunction,
                 tau: float, alpha: float, clip_min: float, clip_max: float):
        self._target_function = target_function
        self._tau = tau
        self._alpha = alpha
        self._clip_min = clip_min
        self._clip_max = clip_max

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        s_current = training_variables.s_current
        a_current = training_variables.a_current
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        N_prime = kwargs['N_prime']

        batch_size = s_next.shape[0]

        tau_j = self._target_function._sample_tau(shape=(batch_size, N_prime))
        target_return_samples = self._target_function.return_samples(s_next, tau_j)
        assert target_return_samples.shape[0:-1] == (batch_size, N_prime)

        all_next_q = self._target_function.return_samples_to_q_values(target_return_samples)
        max_next_q = NF.max(all_next_q, axis=1, keepdims=True)
        pi = _pi(all_next_q, max_next_q, tau=self._tau)
        pi = RNF.expand_dims(pi, axis=1)
        all_tau_log_pi = _all_tau_log_pi(all_next_q, max_next_q, self._tau)
        all_tau_log_pi = RNF.expand_dims(all_tau_log_pi, axis=1)
        assert pi.shape[1] == 1
        assert pi.shape == all_tau_log_pi.shape
        soft_q_target = NF.sum(pi * (target_return_samples - all_tau_log_pi), axis=(pi.ndim - 1))

        current_return_samples = self._target_function.return_samples(s_current, tau_j)
        all_current_q = self._target_function.return_samples_to_q_values(current_return_samples)
        max_current_q = NF.max(all_current_q, axis=1, keepdims=True)
        one_hot = NF.one_hot(NF.reshape(a_current, (-1, 1), inplace=False), (all_current_q.shape[1],))
        current_q = NF.sum(all_current_q * one_hot, axis=1, keepdims=True)  # get q value of a

        tau_log_pi = _tau_log_pi(current_q, all_current_q, max_current_q, self._tau)
        clipped_tau_log_pi = NF.clip_by_value(tau_log_pi, self._clip_min, self._clip_max)

        return reward + self._alpha * clipped_tau_log_pi + gamma * non_terminal * soft_q_target


class MunchausenRLTraining(Training):
    _delegate: Training

    def __init__(self, train_function: Model, target_function: Model,
                 tau: float = 0.03, alpha: float = 0.9, clip_min: float = -1.0, clip_max: float = 0.0):
        if type(train_function) is not type(target_function):
            raise ValueError
        if isinstance(target_function, QFunction):
            self._delegate = _QFunctionMunchausenRLTraining(
                target_function, tau, alpha, clip_min, clip_max)
        elif isinstance(target_function, StateActionQuantileFunction):
            self._delegate = _StateActionQuantileFunctionMunchausenRLTraining(
                target_function, tau, alpha, clip_min, clip_max)
        else:
            raise NotImplementedError(
                f'No training implementation for class: {target_function.__class__}')

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        return self._delegate.compute_target(training_variables, **kwargs)
