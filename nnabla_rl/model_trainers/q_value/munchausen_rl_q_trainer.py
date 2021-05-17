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

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.q_value.squared_td_q_function_trainer import (SquaredTDQFunctionTrainer,
                                                                            SquaredTDQFunctionTrainerConfig)
from nnabla_rl.model_trainers.q_value.state_action_quantile_function_trainer import (
    StateActionQuantileFunctionTrainer, StateActionQuantileFunctionTrainerConfig)
from nnabla_rl.models import QFunction, StateActionQuantileFunction


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


@dataclass
class MunchausenDQNQTrainerConfig(SquaredTDQFunctionTrainerConfig):
    tau: float = 0.03
    alpha: float = 0.9
    clip_min: float = -1.0
    clip_max: float = 0.0


class MunchausenDQNQTrainer(SquaredTDQFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: QFunction
    _config: MunchausenDQNQTrainerConfig

    def __init__(self,
                 train_functions: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: MunchausenDQNQTrainerConfig = MunchausenDQNQTrainerConfig()):
        self._target_function = target_function
        super(MunchausenDQNQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        s_current = training_variables.s_current
        a_current = training_variables.a_current
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        all_next_q = self._target_function.all_q(s_next)
        max_next_q = self._target_function.max_q(s_next)
        pi = _pi(all_next_q, max_next_q, tau=self._config.tau)
        all_tau_log_pi = _all_tau_log_pi(all_next_q, max_next_q, self._config.tau)
        assert pi.shape == all_next_q.shape
        assert pi.shape == all_tau_log_pi.shape
        soft_q_target = NF.sum(pi * (all_next_q - all_tau_log_pi), axis=(pi.ndim - 1),  keepdims=True)

        current_q = self._target_function.q(s_current, a_current)
        all_current_q = self._target_function.all_q(s_current)
        max_current_q = self._target_function.max_q(s_current)
        tau_log_pi = _tau_log_pi(current_q, all_current_q, max_current_q, self._config.tau)
        clipped_tau_log_pi = NF.clip_by_value(tau_log_pi, self._config.clip_min, self._config.clip_max)
        return reward + self._config.alpha * clipped_tau_log_pi + gamma * non_terminal * soft_q_target


@dataclass
class MunchausenIQNQTrainerConfig(StateActionQuantileFunctionTrainerConfig):
    tau: float = 0.03
    alpha: float = 0.9
    clip_min: float = -1.0
    clip_max: float = 0.0


class MunchausenIQNQTrainer(StateActionQuantileFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: StateActionQuantileFunction
    _config: MunchausenIQNQTrainerConfig

    def __init__(self,
                 train_functions: Union[StateActionQuantileFunction, Sequence[StateActionQuantileFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_function: StateActionQuantileFunction,
                 env_info: EnvironmentInfo,
                 config: MunchausenIQNQTrainerConfig = MunchausenIQNQTrainerConfig()):
        self._target_function = target_function
        super(MunchausenIQNQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        s_current = training_variables.s_current
        a_current = training_variables.a_current
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        batch_size = s_next.shape[0]

        tau_j = self._target_function.sample_tau(shape=(batch_size, self._config.N_prime))
        target_return_samples = self._target_function.all_quantile_values(s_next, tau_j)
        assert target_return_samples.shape[0:-1] == (batch_size, self._config.N_prime)

        all_next_q = NF.transpose(target_return_samples, axes=(0, 2, 1))
        all_next_q = NF.mean(all_next_q, axis=2)
        max_next_q = NF.max(all_next_q, axis=1, keepdims=True)
        pi = _pi(all_next_q, max_next_q, tau=self._config.tau)
        pi = RF.expand_dims(pi, axis=1)
        all_tau_log_pi = _all_tau_log_pi(all_next_q, max_next_q, self._config.tau)
        all_tau_log_pi = RF.expand_dims(all_tau_log_pi, axis=1)
        assert pi.shape[1] == 1
        assert pi.shape == all_tau_log_pi.shape
        soft_q_target = NF.sum(pi * (target_return_samples - all_tau_log_pi), axis=(pi.ndim - 1))

        current_return_samples = self._target_function.all_quantile_values(s_current, tau_j)
        all_current_q = NF.transpose(current_return_samples, axes=(0, 2, 1))
        all_current_q = NF.mean(all_current_q, axis=2)
        max_current_q = NF.max(all_current_q, axis=1, keepdims=True)
        one_hot = NF.one_hot(NF.reshape(a_current, (-1, 1), inplace=False), (all_current_q.shape[1],))
        current_q = NF.sum(all_current_q * one_hot, axis=1, keepdims=True)  # get q value of a

        tau_log_pi = _tau_log_pi(current_q, all_current_q, max_current_q, self._config.tau)
        clipped_tau_log_pi = NF.clip_by_value(tau_log_pi, self._config.clip_min, self._config.clip_max)

        return reward + self._config.alpha * clipped_tau_log_pi + gamma * non_terminal * soft_q_target
