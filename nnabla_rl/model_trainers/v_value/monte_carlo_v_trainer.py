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
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.v_value.squared_td_v_function_trainer import (SquaredTDVFunctionTrainer,
                                                                            SquaredTDVFunctionTrainerConfig)
from nnabla_rl.models import VFunction


@dataclass
class MonteCarloVTrainerConfig(SquaredTDVFunctionTrainerConfig):
    pass


class MonteCarloVTrainer(SquaredTDVFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _v_target: nn.Variable

    def __init__(self,
                 train_functions: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: MonteCarloVTrainerConfig = MonteCarloVTrainerConfig()):
        super(MonteCarloVTrainer, self).__init__(train_functions, solvers, env_info, config)

    def _setup_batch(self, batch: TrainingBatch):
        batch_size = batch.batch_size
        v_target = batch.extra['v_target']
        if self._v_target is None or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        self._v_target.d = v_target
        return batch

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        if not hasattr(self, '_v_target') or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        return self._v_target
