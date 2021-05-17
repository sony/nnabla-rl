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

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import DeterministicPolicy, Model, QFunction


@dataclass
class DPGPolicyTrainerConfig(TrainerConfig):
    pass


class DPGPolicyTrainer(ModelTrainer):
    '''Deterministic Policy Gradient (DPG) style Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DPGPolicyTrainerConfig
    _pi_loss: nn.Variable
    _q_function: QFunction

    def __init__(self,
                 models: Union[DeterministicPolicy, Sequence[DeterministicPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: DPGPolicyTrainerConfig = DPGPolicyTrainerConfig()):
        self._q_function = q_function
        super(DPGPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = float(self._pi_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DeterministicPolicy], models)
        self._pi_loss = 0
        for policy in models:
            action = policy.pi(training_variables.s_current)
            q = self._q_function.q(training_variables.s_current, action)
            self._pi_loss += -NF.mean(q)

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        return TrainingVariables(batch_size, s_current_var)
