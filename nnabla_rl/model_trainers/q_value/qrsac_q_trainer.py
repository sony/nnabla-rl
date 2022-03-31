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

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.quantile_distribution_function_trainer import (
    QuantileDistributionFunctionTrainer, QuantileDistributionFunctionTrainerConfig)
from nnabla_rl.models import QuantileDistributionFunction
from nnabla_rl.models.policy import StochasticPolicy
from nnabla_rl.utils.data import convert_to_list_if_not_list
from nnabla_rl.utils.misc import create_variables


@dataclass
class QRSACQTrainerConfig(QuantileDistributionFunctionTrainerConfig):
    pass


class QRSACQTrainer(QuantileDistributionFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_function: QuantileDistributionFunction
    _prev_target_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_quantile_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 train_functions: Union[QuantileDistributionFunction, Sequence[QuantileDistributionFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[QuantileDistributionFunction, Sequence[QuantileDistributionFunction]],
                 target_policy: StochasticPolicy,
                 temperature: nn.Variable,
                 env_info: EnvironmentInfo,
                 config: QRSACQTrainerConfig = QRSACQTrainerConfig()):
        self._target_functions = convert_to_list_if_not_list(target_functions)
        self._assert_no_duplicate_model(self._target_functions)
        self._target_policy = target_policy
        self._temperature = temperature
        self._prev_target_rnn_states = {}
        self._prev_quantile_rnn_states = {}
        self._prev_q_rnn_states = {}
        super(QRSACQTrainer, self).__init__(train_functions, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        train_rnn_states = training_variables.rnn_states
        prev_rnn_states = self._prev_target_rnn_states
        with rnn_support(self._target_policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
            policy_distribution = self._target_policy.pi(s_next)
        a_next, log_pi = policy_distribution.sample_and_compute_log_prob()

        q_list: List[nn.Variable] = []
        prev_rnn_states = self._prev_q_rnn_states
        for target_function in self._target_functions:
            with rnn_support(target_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_function = target_function.as_q_function()
                q = q_function.q(s_next, a_next)
                q_list.append(q)

        qs = NF.concatenate(*q_list, axis=len(q_list[0].shape) - 1)
        min_indices = RF.argmin(qs, axis=len(qs.shape) - 1)

        theta_js = []
        prev_rnn_states = self._prev_quantile_rnn_states
        for target_quantile_function in self._target_functions:
            with rnn_support(target_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                theta_j = target_quantile_function.quantiles(s_next, a_next)
                theta_js.append(theta_j)

        theta_js = NF.stack(*theta_js, axis=len(theta_js[0].shape) - 1)
        theta_j = NF.gather(theta_js, min_indices, axis=1, batch_dims=1)

        Ttheta_j = reward + gamma * non_terminal * (theta_j - self._temperature * log_pi)
        return Ttheta_j

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        rnn_states = {}
        for target_function in self._target_functions:
            if target_function.is_recurrent():
                shapes = target_function.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[target_function.scope_name] = rnn_state_variables
        if self._target_policy.is_recurrent():
            shapes = self._target_policy.internal_state_shapes()
            rnn_state_variables = create_variables(batch_size, shapes)
            rnn_states[self._target_policy.scope_name] = rnn_state_variables

        training_variables.rnn_states.update(rnn_states)
        return training_variables
