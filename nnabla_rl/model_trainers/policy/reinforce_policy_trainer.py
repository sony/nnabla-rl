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
from nnabla_rl.models.model import Model
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


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

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: REINFORCEPolicyTrainerConfig = REINFORCEPolicyTrainerConfig()):
        super(REINFORCEPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.extra['target_return'], b.extra['target_return'])
        return super()._update_model(models, solvers, batch, training_variables, **kwargs)

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        return training_variables.extra['target_return']

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        extra = {}
        extra['target_return'] = create_variable(batch_size, 1)
        training_variables.extra.update(extra)

        return training_variables
