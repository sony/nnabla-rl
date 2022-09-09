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
from typing import Dict, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables, rnn_support)
from nnabla_rl.models import DeterministicPolicy, Model, QFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables


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
    _prev_policy_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rnn_states: Dict[str, Dict[str, Dict[str, nn.Variable]]]

    def __init__(self,
                 models: Union[DeterministicPolicy, Sequence[DeterministicPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_function: QFunction,
                 env_info: EnvironmentInfo,
                 config: DPGPolicyTrainerConfig = DPGPolicyTrainerConfig()):
        self._q_function = q_function
        self._prev_policy_rnn_states = {}
        self._prev_q_rnn_states = {}
        for model in convert_to_list_if_not_list(models):
            self._prev_q_rnn_states[model.scope_name] = {}

        super(DPGPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return True

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
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
            if self._q_function.is_recurrent() and self._q_function.scope_name in batch.rnn_states.keys():
                b_rnn_states = b.rnn_states[self._q_function.scope_name]
                t_rnn_states = t.rnn_states[self._q_function.scope_name]
                for state_name in t_rnn_states.keys():
                    set_data_to_variable(t_rnn_states[state_name], b_rnn_states[state_name])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = self._pi_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DeterministicPolicy], models)
        self._pi_loss = 0
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
        models = cast(Sequence[DeterministicPolicy], models)
        train_rnn_states = training_variables.rnn_states
        for policy in models:
            prev_rnn_states = self._prev_policy_rnn_states
            with rnn_support(policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
                action = policy.pi(training_variables.s_current)

            prev_rnn_states = self._prev_q_rnn_states[policy.scope_name]
            with rnn_support(self._q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q = self._q_function.q(training_variables.s_current, action)
            self._prev_q_rnn_states[policy.scope_name] = prev_rnn_states
            self._pi_loss += 0.0 if ignore_loss else -NF.mean(q)

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        non_terminal_var = create_variable(batch_size, 1)

        rnn_states = {}
        for policy in self._models:
            if policy.is_recurrent():
                shapes = policy.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[policy.scope_name] = rnn_state_variables
        if self._q_function.is_recurrent():
            shapes = self._q_function.internal_state_shapes()
            rnn_state_variables = create_variables(batch_size, shapes)
            rnn_states[self._q_function.scope_name] = rnn_state_variables

        return TrainingVariables(batch_size, s_current_var, non_terminal=non_terminal_var, rnn_states=rnn_states)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}
