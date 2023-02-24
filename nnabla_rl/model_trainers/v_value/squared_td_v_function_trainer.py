# Copyright 2021,2022,2023 Sony Group Corporation.
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
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.v_value.v_function_trainer import VFunctionTrainer, VFunctionTrainerConfig
from nnabla_rl.models import VFunction


@dataclass
class SquaredTDVFunctionTrainerConfig(VFunctionTrainerConfig):
    pass


class SquaredTDVFunctionTrainer(VFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: SquaredTDVFunctionTrainerConfig
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SquaredTDVFunctionTrainerConfig = SquaredTDVFunctionTrainerConfig()):
        self._prev_rnn_states = {}
        super(SquaredTDVFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _compute_loss(self,
                      model: VFunction,
                      target_value: nn.Variable,
                      training_variables: TrainingVariables) -> nn.Variable:
        prev_rnn_states = self._prev_rnn_states
        train_rnn_states = training_variables.rnn_states
        with rnn_support(model, prev_rnn_states, train_rnn_states, training_variables, self._config):
            v_value = model.v(training_variables.s_current)
        td_error = target_value - v_value
        return NF.pow_scalar(td_error, 2.0)
