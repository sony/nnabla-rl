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

import numpy as np

import nnabla as nn
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.policy.spg_policy_trainer import SPGPolicyTrainer, SPGPolicyTrainerConfig
from nnabla_rl.models import StochasticPolicy


@dataclass
class REINFORCEPolicyTrainerConfig(SPGPolicyTrainerConfig):
    pass


class REINFORCEPolicyTrainer(SPGPolicyTrainer):
    '''REINFORCE style Stochastic Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: REINFORCEPolicyTrainerConfig
    _target_return: nn.Variable

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: REINFORCEPolicyTrainerConfig = REINFORCEPolicyTrainerConfig()):
        super(REINFORCEPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _setup_batch(self, batch: TrainingBatch):
        target_return = batch.extra['target_return']
        prev_batch_size = self._target_return.shape[0]
        new_batch_size = target_return.shape[0]
        if prev_batch_size != new_batch_size:
            self._target_return = nn.Variable((new_batch_size, 1))
        target_return = np.reshape(target_return, self._target_return.shape)
        self._target_return.d = target_return
        return batch

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        batch_size = training_variables.batch_size
        if not hasattr(self, '_target_return') or self._target_return.shape[0] != batch_size:
            self._target_return = nn.Variable((batch_size, 1))
        return self._target_return
