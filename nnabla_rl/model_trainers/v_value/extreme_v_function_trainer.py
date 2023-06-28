# Copyright 2023 Sony Group Corporation.
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
from typing import Dict, Optional, Sequence, Union

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.v_value.v_function_trainer import VFunctionTrainer, VFunctionTrainerConfig
from nnabla_rl.models import VFunction


@dataclass
class ExtremeVFunctionTrainerConfig(VFunctionTrainerConfig):
    beta: float = 1.0
    max_clip: Optional[float] = 7.0

    def __post_init__(self):
        super(ExtremeVFunctionTrainerConfig, self).__post_init__()
        self._assert_positive(self.beta, 'beta')


class ExtremeVFunctionTrainer(VFunctionTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ExtremeVFunctionTrainerConfig
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: ExtremeVFunctionTrainerConfig = ExtremeVFunctionTrainerConfig()):
        self._prev_rnn_states = {}
        super(ExtremeVFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _compute_loss(self,
                      model: VFunction,
                      target_value: nn.Variable,
                      training_variables: TrainingVariables) -> nn.Variable:
        prev_rnn_states = self._prev_rnn_states
        train_rnn_states = training_variables.rnn_states
        with rnn_support(model, prev_rnn_states, train_rnn_states, training_variables, self._config):
            v_value = model.v(training_variables.s_current)
        z = (target_value - v_value) / self._config.beta
        if self._config.max_clip is not None:
            z = NF.minimum_scalar(z, self._config.max_clip)
        max_z = NF.max(z, axis=0, keepdims=True)
        max_z = NF.maximum_scalar(max_z, val=-1.0)
        max_z.need_grad = False
        # original code seems to rescale the gumbel loss by max(z)
        # i.e. exp(z) / max(z) - z / max(z)
        # - NF.exp(-max_z) <- this term exists in the original code but this should take no effect
        return NF.exp(z-max_z) - z * NF.exp(-max_z) - NF.exp(-max_z)
