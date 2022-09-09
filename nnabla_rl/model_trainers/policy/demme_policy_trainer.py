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
from typing import Dict, Sequence, Union

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables, rnn_support)
from nnabla_rl.models import Model, QFunction, StochasticPolicy
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables


@dataclass
class DEMMEPolicyTrainerConfig(TrainerConfig):
    def __post_init__(self):
        super(DEMMEPolicyTrainerConfig, self).__post_init__()


class DEMMEPolicyTrainer(ModelTrainer):
    '''DEMME Policy Gradient style Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _q_rr_functions: Sequence[QFunction]
    _q_re_functions: Sequence[QFunction]
    _config: DEMMEPolicyTrainerConfig
    _pi_loss: nn.Variable
    _prev_policy_rnn_states: Dict[str, Dict[str, nn.Variable]]
    _prev_q_rr_rnn_states: Dict[str, Dict[str, Dict[str, nn.Variable]]]
    _prev_q_re_rnn_states: Dict[str, Dict[str, Dict[str, nn.Variable]]]

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_rr_functions: Sequence[QFunction],
                 q_re_functions: Sequence[QFunction],
                 env_info: EnvironmentInfo,
                 config: DEMMEPolicyTrainerConfig = DEMMEPolicyTrainerConfig()):
        if len(q_rr_functions) < 2:
            raise ValueError('Must provide at least 2 Qrr-functions for DEMME-training')
        if len(q_re_functions) < 2:
            raise ValueError('Must provide at least 2 Qre-functions for DEMME-training')
        self._q_rr_functions = q_rr_functions
        self._q_re_functions = q_re_functions

        self._prev_policy_rnn_states = {}
        self._prev_q_rr_rnn_states = {}
        self._prev_q_re_rnn_states = {}
        for model in convert_to_list_if_not_list(models):
            self._prev_q_rr_rnn_states[model.scope_name] = {}
            self._prev_q_re_rnn_states[model.scope_name] = {}

        super(DEMMEPolicyTrainer, self).__init__(models, solvers, env_info, config)

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
            for q_function in self._q_rr_functions:
                if not q_function.is_recurrent():
                    continue
                # Check batch keys. Because it can be empty.
                # If batch does not provide rnn states, train with zero initial state.
                if q_function.scope_name not in batch.rnn_states.keys():
                    continue
                b_rnn_states = b.rnn_states[q_function.scope_name]
                t_rnn_states = t.rnn_states[q_function.scope_name]

                for state_name in t_rnn_states.keys():
                    set_data_to_variable(t_rnn_states[state_name], b_rnn_states[state_name])
            for q_function in self._q_re_functions:
                if not q_function.is_recurrent():
                    continue
                # Check batch keys. Because it can be empty.
                # If batch does not provide rnn states, train with zero initial state.
                if q_function.scope_name not in batch.rnn_states.keys():
                    continue
                b_rnn_states = b.rnn_states[q_function.scope_name]
                t_rnn_states = t.rnn_states[q_function.scope_name]

                for state_name in t_rnn_states.keys():
                    set_data_to_variable(t_rnn_states[state_name], b_rnn_states[state_name])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward()
        self._pi_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = self._pi_loss.d.copy()
        return trainer_state

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
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
        train_rnn_states = training_variables.rnn_states
        for policy in models:
            assert isinstance(policy, StochasticPolicy)
            # Actor optimization graph
            prev_rnn_states = self._prev_policy_rnn_states
            with rnn_support(policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
                policy_distribution = policy.pi(training_variables.s_current)
            action_var, log_pi = policy_distribution.sample_and_compute_log_prob()

            # Q_rr
            q_values = []
            prev_rnn_states = self._prev_q_rr_rnn_states[policy.scope_name]
            for q_function in self._q_rr_functions:
                with rnn_support(q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                    q_values.append(q_function.q(training_variables.s_current, action_var))
            self._prev_q_rr_rnn_states[policy.scope_name] = prev_rnn_states
            min_q_rr = RNF.minimum_n(q_values)

            # Q_re
            q_values = []
            prev_rnn_states = self._prev_q_re_rnn_states[policy.scope_name]
            for q_function in self._q_re_functions:
                with rnn_support(q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                    q_values.append(q_function.q(training_variables.s_current, action_var))
            self._prev_q_re_rnn_states[policy.scope_name] = prev_rnn_states
            min_q_re = RNF.minimum_n(q_values)

            self._pi_loss += NF.mean(log_pi - min_q_rr - min_q_re)

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

        for q_function in self._q_rr_functions:
            if q_function.is_recurrent():
                shapes = q_function.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[q_function.scope_name] = rnn_state_variables

        for q_function in self._q_re_functions:
            if q_function.is_recurrent():
                shapes = q_function.internal_state_shapes()
                rnn_state_variables = create_variables(batch_size, shapes)
                rnn_states[q_function.scope_name] = rnn_state_variables

        return TrainingVariables(batch_size, s_current_var, non_terminal=non_terminal_var, rnn_states=rnn_states)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}
