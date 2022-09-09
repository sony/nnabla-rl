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
from typing import Dict, Optional, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables, rnn_support)
from nnabla_rl.models import Model, VFunction
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import clip_grad_by_global_norm, create_variable, create_variables


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
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[VFunction, Sequence[VFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SquaredTDVFunctionTrainerConfig = SquaredTDVFunctionTrainerConfig()):
        self._prev_rnn_states = {}
        super(SquaredTDVFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            if self.support_rnn() and self._config.reset_on_terminal and self._need_rnn_support(models):
                set_data_to_variable(t.non_terminal, b.non_terminal)

            for model in models:
                if not model.is_recurrent():
                    continue
                # Check batch keys. Because it can be empty.
                # If batch does not provide rnn states, train with zero initial state.
                if model.scope_name not in batch.rnn_states.keys():
                    continue
                b_rnn_states = b.rnn_states[model.scope_name]
                t_rnn_states = t.rnn_states[model.scope_name]

                for state_name in t_rnn_states.keys():
                    set_data_to_variable(t_rnn_states[state_name], b_rnn_states[state_name])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._v_loss.forward(clear_no_need_grad=True)
        self._v_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            if self._config.max_grad_norm is not None:
                clip_grad_by_global_norm(solver, self._config.max_grad_norm)
            solver.update()

        trainer_state: Dict[str, np.ndarray] = {}
        trainer_state['v_loss'] = self._v_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Union[Model, Sequence[Model]], training_variables: TrainingVariables):
        self._v_loss = 0
        models = cast(Sequence[VFunction], models)
        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables,
                              ignore_loss: bool):
        # value function learning
        target_v = self._compute_target(training_variables)
        target_v.need_grad = False

        prev_rnn_states = self._prev_rnn_states
        train_rnn_states = training_variables.rnn_states
        for v_function in models:
            v_function = cast(VFunction, v_function)
            with rnn_support(v_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                v_loss = self._compute_loss(v_function, target_v, training_variables)
            self._v_loss += 0.0 if ignore_loss else v_loss

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
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        # Only used with rnn training
        non_terminal_var = create_variable(batch_size, 1)

        rnn_states = {}
        for model in self._models:
            if model.is_recurrent():
                rnn_state_variables = create_variables(batch_size, model.internal_state_shapes())
                rnn_states[model.scope_name] = rnn_state_variables

        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               non_terminal=non_terminal_var,
                                               rnn_states=rnn_states)
        return training_variables

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"v_loss": self._v_loss}
