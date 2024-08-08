# Copyright 2023,2024 Sony Group Corporation.
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
from typing import Any, Dict, List, Sequence, Tuple, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import LossIntegration, TrainingBatch, TrainingVariables, rnn_support
from nnabla_rl.model_trainers.policy import DPGPolicyTrainer, DPGPolicyTrainerConfig
from nnabla_rl.models import DeterministicPolicy, Model, QFunction
from nnabla_rl.utils.data import set_data_to_variable


@dataclass
class HyARPolicyTrainerConfig(DPGPolicyTrainerConfig):
    p_min: Union[np.ndarray, None] = None
    p_max: Union[np.ndarray, None] = None


class HyARPolicyTrainer(DPGPolicyTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _grads_sum: nn.Variable
    _config: HyARPolicyTrainerConfig
    _action_and_grads: Dict[str, List[Tuple[nn.Variable, nn.Variable]]]

    def __init__(
        self,
        models: Union[DeterministicPolicy, Sequence[DeterministicPolicy]],
        solvers: Dict[str, nn.solver.Solver],
        q_function: QFunction,
        env_info: EnvironmentInfo,
        config: HyARPolicyTrainerConfig = HyARPolicyTrainerConfig(),
    ):
        super().__init__(models, solvers, q_function, env_info, config)

    def _update_model(
        self,
        models: Sequence[Model],
        solvers: Dict[str, nn.solver.Solver],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
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
        self._grads_sum.forward()
        for solver in solvers.values():
            solver.zero_grad()
        for action_grad_list in self._action_and_grads.values():
            for action, grad in action_grad_list:
                action.backward(grad=-grad.d)
        for solver in solvers.values():
            solver.update()

        trainer_state: Dict[str, Any] = {"pi_loss": 0}
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DeterministicPolicy], models)
        self._action_and_grads = {policy.scope_name: [] for policy in models}
        self._grads_sum = 0
        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        models = cast(Sequence[DeterministicPolicy], models)
        train_rnn_states = training_variables.rnn_states
        for policy in models:
            prev_rnn_states = self._prev_policy_rnn_states
            with rnn_support(policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
                action = policy.pi(training_variables.s_current)

            prev_rnn_states = self._prev_q_rnn_states[policy.scope_name]
            with rnn_support(self._q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                action_grad = self._compute_q_grad_wrt_action(training_variables.s_current, action, self._q_function)
            action_grad = self._invert_gradients(action, action_grad, self._p_min, self._p_max)
            action_grad.persistent = True
            self._prev_q_rnn_states[policy.scope_name] = prev_rnn_states
            if not ignore_loss:
                self._action_and_grads[policy.scope_name].append((action, action_grad))
                self._grads_sum += action_grad

    def _compute_q_grad_wrt_action(self, state: nn.Variable, action: nn.Variable, q_function: QFunction):
        q = NF.mean(q_function.q(state, action))
        return nn.grad(outputs=[q], inputs=action)[0]

    def _invert_gradients(self, p: nn.Variable, grads: nn.Variable, p_min: nn.Variable, p_max: nn.Variable):
        increasing = NF.greater_equal_scalar(grads, val=0)
        decreasing = NF.less_scalar(grads, val=0)
        p_range = p_max - p_min
        return grads * increasing * (p_max - p) / p_range + grads * decreasing * (p - p_min) / p_range

    @property
    def _p_min(self) -> nn.Variable:
        return nn.Variable.from_numpy_array(self._config.p_min)

    @property
    def _p_max(self) -> nn.Variable:
        return nn.Variable.from_numpy_array(self._config.p_max)

    def support_rnn(self) -> bool:
        # TODO: support rnn
        return False
