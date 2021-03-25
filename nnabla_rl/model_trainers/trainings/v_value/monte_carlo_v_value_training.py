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
from nnabla_rl.model_trainers.model_trainer import Training, TrainingBatch, TrainingVariables


class MonteCarloVValueTraining(Training):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _v_target: nn.Variable

    def __init__(self):
        super(MonteCarloVValueTraining, self).__init__()

    def setup_batch(self, batch: TrainingBatch):
        batch_size = batch.batch_size
        v_target = batch.extra['v_target']
        if self._v_target is None or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        self._v_target.d = v_target
        return batch

    def compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        batch_size = training_variables.batch_size
        if not hasattr(self, '_v_target') or self._v_target.shape[0] != batch_size:
            self._v_target = nn.Variable((batch_size, 1))
        return self._v_target
