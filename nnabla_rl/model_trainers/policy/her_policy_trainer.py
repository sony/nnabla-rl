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
from typing import Dict, Sequence, Union, cast

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables
from nnabla_rl.model_trainers.policy import DPGPolicyTrainer, DPGPolicyTrainerConfig
from nnabla_rl.models import DeterministicPolicy, Model, QFunction


@dataclass
class HERPolicyTrainerConfig(DPGPolicyTrainerConfig):
    action_loss_coef: float = 1.0


class HERPolicyTrainer(DPGPolicyTrainer):
    _config: HERPolicyTrainerConfig

    def __init__(self,
                 models: Union[DeterministicPolicy, Sequence[DeterministicPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: HERPolicyTrainerConfig = HERPolicyTrainerConfig()):
        self._max_action_value = float(env_info.action_space.high[0])
        super(HERPolicyTrainer, self).__init__(models, solvers, q_function, env_info, config)

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DeterministicPolicy], models)
        self._pi_loss = 0
        for policy in models:
            action = policy.pi(training_variables.s_current)
            q = self._q_function.q(training_variables.s_current, action)
            self._pi_loss += -NF.mean(q)
            self._pi_loss += self._config.action_loss_coef \
                * NF.mean(NF.pow_scalar(action / self._max_action_value, 2.0))
