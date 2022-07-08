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

from dataclasses import dataclass
from typing import Dict, Sequence, Union, cast

import gym

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.policy import DPGPolicyTrainer, DPGPolicyTrainerConfig
from nnabla_rl.models import DeterministicPolicy, Model, QFunction


@dataclass
class HERPolicyTrainerConfig(DPGPolicyTrainerConfig):
    action_loss_coef: float = 1.0


class HERPolicyTrainer(DPGPolicyTrainer):
    _config: HERPolicyTrainerConfig
    _train_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[DeterministicPolicy, Sequence[DeterministicPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: HERPolicyTrainerConfig = HERPolicyTrainerConfig()):
        action_space = cast(gym.spaces.Box, env_info.action_space)
        self._max_action_value = float(action_space.high[0])
        super(HERPolicyTrainer, self).__init__(models, solvers, q_function, env_info, config)

    def _build_one_step_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables,
                              ignore_loss: bool):
        models = cast(Sequence[DeterministicPolicy], models)
        train_rnn_states = training_variables.rnn_states
        for policy in models:
            prev_rnn_states = self._prev_policy_rnn_states
            with rnn_support(policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
                action = policy.pi(training_variables.s_current)

            prev_rnn_states = self._prev_q_rnn_states[policy.scope_name]
            with rnn_support(self._q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q = self._q_function.q(training_variables.s_current, action)
            self._prev_q_rnn_states[policy.scope_name] = prev_rnn_states

            self._pi_loss += 0.0 if ignore_loss else -NF.mean(q)
            self._pi_loss += 0.0 if ignore_loss else self._config.action_loss_coef \
                * NF.mean(NF.pow_scalar(action / self._max_action_value, 2.0))
