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
from typing import Dict, Optional, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, VFunction
from nnabla_rl.utils.misc import clip_grad_by_global_norm


@dataclass
class SquaredTDVFunctionTrainerConfig(TrainerConfig):
    reduction_method: str = 'mean'
    v_loss_scalar: float = 1.0
    max_grad_norm: Optional[float] = None

    def __post_init__(self):
        super(SquaredTDVFunctionTrainerConfig, self).__post_init__()
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')


class SquaredTDVFunctionTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: SquaredTDVFunctionTrainerConfig
    _v_loss: nn.Variable  # Training loss/output

    def __init__(self,
                 models: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SquaredTDVFunctionTrainerConfig = SquaredTDVFunctionTrainerConfig()):
        super(SquaredTDVFunctionTrainer, self).__init__(models, solvers, env_info, config)

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
        self._v_loss.forward(clear_no_need_grad=True)
        self._v_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            if self._config.max_grad_norm is not None:
                clip_grad_by_global_norm(solver, self._config.max_grad_norm)
            solver.update()

        trainer_state = {}
        trainer_state['v_loss'] = float(self._v_loss.d.copy())
        return trainer_state

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        models = cast(Sequence[VFunction], models)

        # value function learning
        target_v = self._compute_target(training_variables)

        self._v_loss = 0
        for v_function in models:
            self._v_loss += self._compute_loss(v_function, target_v, training_variables)

    def _compute_loss(self,
                      model: VFunction,
                      target_value: nn.Variable,
                      training_variables: TrainingVariables) -> nn.Variable:
        td_error = target_value - model.v(training_variables.s_current)
        squared_td_error = NF.pow_scalar(td_error, 2.0)
        if self._config.reduction_method == 'mean':
            v_loss = self._config.v_loss_scalar * NF.mean(squared_td_error)
        elif self._config.reduction_method == 'sum':
            v_loss = self._config.v_loss_scalar * NF.sum(squared_td_error)
        else:
            raise RuntimeError
        return v_loss

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        raise NotImplementedError

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        training_variables = TrainingVariables(batch_size, s_current_var)
        return training_variables
