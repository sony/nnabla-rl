# Copyright 2022 Sony Group Corporation.
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
from typing import Dict, Iterable, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla_rl.functions as RF
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables, rnn_support)
from nnabla_rl.models import DeterministicDynamics, Model
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables


@dataclass
class MPPIDynamicsTrainerConfig(TrainerConfig):
    dt: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self._assert_positive(self.dt, 'dt')


class MPPIDynamicsTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: MPPIDynamicsTrainerConfig
    _loss: nn.Variable
    _prev_dynamics_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[DeterministicDynamics, Sequence[DeterministicDynamics]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info,
                 config: MPPIDynamicsTrainerConfig = MPPIDynamicsTrainerConfig()):
        self._prev_dynamics_rnn_states = {}
        super(MPPIDynamicsTrainer, self).__init__(models, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.s_next, b.s_next)
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
        self._loss.forward(clear_no_need_grad=True)
        self._loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state: Dict[str, np.ndarray] = {}
        trainer_state['dynamics_loss'] = self._loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Union[Model, Sequence[Model]],
                              training_variables: TrainingVariables):
        models = convert_to_list_if_not_list(models)
        models = cast(Sequence[DeterministicDynamics], models)

        self._loss = 0
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
        models = cast(Sequence[DeterministicDynamics], models)
        train_rnn_states = training_variables.rnn_states
        for model in models:
            prev_rnn_states = self._prev_dynamics_rnn_states
            with rnn_support(model, prev_rnn_states, train_rnn_states, training_variables, self._config):
                predicted_a = model.acceleration(training_variables.s_current, training_variables.a_current)

            s_current = cast(nn.Variable, training_variables.s_current)
            s_next = cast(nn.Variable, training_variables.s_next)
            q_dot = s_current[:, self._env_info.state_dim // 2:]
            q_dot_next = s_next[:, self._env_info.state_dim // 2:]
            target_a = (q_dot_next - q_dot) / self._config.dt
            target_a.need_grad = False
            loss = RF.mean_squared_error(predicted_a, target_a)
            self._loss += 0.0 if ignore_loss else loss

    def _setup_training_variables(self, batch_size):
        s_current_var = create_variable(batch_size, self._env_info.state_dim)
        a_current_var = create_variable(batch_size, self._env_info.action_dim)
        non_terminal_var = create_variable(batch_size, 1)
        s_next_var = create_variable(batch_size, self._env_info.state_dim)

        rnn_states = {}
        for model in self._models:
            if model.is_recurrent():
                rnn_state_variables = create_variables(batch_size, model.internal_state_shapes())
                rnn_states[model.scope_name] = rnn_state_variables

        return TrainingVariables(batch_size, s_current_var, a_current_var, s_next=s_next_var,
                                 non_terminal=non_terminal_var,
                                 rnn_states=rnn_states)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"loss": self._loss}
