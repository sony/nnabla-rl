# Copyright 2024 Sony Group Corporation.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (
    LossIntegration,
    ModelTrainer,
    TrainerConfig,
    TrainingBatch,
    TrainingVariables,
)
from nnabla_rl.models import Model, RewardFunction
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class AMPRewardFunctionTrainerConfig(TrainerConfig):
    batch_size: int = 256
    extra_regularization_variable_names: Tuple[str] = ("logits/affine/W",)
    extra_regularization_coefficient: float = 0.05
    regularization_coefficient: float = 0.0005
    gradient_penelty_coefficient: float = 10.0
    gradient_penalty_indexes: Optional[Tuple[int, ...]] = (1,)


class AMPRewardFunctionTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: AMPRewardFunctionTrainerConfig
    _reward_loss: nn.Variable
    _total_reward_loss: nn.Variable
    _binary_regression_loss: nn.Variable
    _extra_regularization_loss: nn.Variable
    _grad_penalty_loss: nn.Variable
    _regularization_loss: nn.Variable

    def __init__(
        self,
        models: Union[RewardFunction, Sequence[RewardFunction]],
        solvers: Dict[str, nn.solver.Solver],
        env_info: EnvironmentInfo,
        state_preprocessor: Optional[Preprocessor] = None,
        config: AMPRewardFunctionTrainerConfig = AMPRewardFunctionTrainerConfig(),
    ):
        self._state_preprocessor = state_preprocessor
        super(AMPRewardFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(
        self,
        models: Iterable[Model],
        solvers: Dict[str, nn.solver.Solver],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            for key in batch.extra.keys():
                set_data_to_variable(t.extra[key], b.extra[key])

        for solver in solvers.values():
            solver.zero_grad()

        self._total_reward_loss.forward()
        self._total_reward_loss.backward()

        for solver in solvers.values():
            solver.update()

        trainer_state: Dict[str, np.ndarray] = {}
        trainer_state["reward_loss"] = self._total_reward_loss.d.copy()
        trainer_state["binary_regression_loss"] = self._binary_regression_loss.d.copy()
        trainer_state["extra_regularization_loss"] = self._extra_regularization_loss.d.copy()
        trainer_state["grad_penalty_loss"] = self._grad_penalty_loss.d.copy()
        trainer_state["regularization_loss"] = self._regularization_loss.d.copy()

        return trainer_state

    def _build_training_graph(
        self,
        models: Union[Model, Sequence[Model]],
        training_variables: TrainingVariables,
    ):
        models = convert_to_list_if_not_list(models)
        models = cast(Sequence[RewardFunction], models)

        self._binary_regression_loss = 0.0
        self._extra_regularization_loss = 0.0
        self._grad_penalty_loss = 0.0
        self._regularization_loss = 0.0

        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

        self._total_reward_loss = (
            self._binary_regression_loss
            + self._grad_penalty_loss
            + self._regularization_loss
            + self._extra_regularization_loss
        )

        # To check all loss is built
        assert isinstance(self._binary_regression_loss, nn.Variable)
        assert isinstance(self._grad_penalty_loss, nn.Variable)
        assert isinstance(self._regularization_loss, nn.Variable)
        assert isinstance(self._extra_regularization_loss, nn.Variable)
        assert isinstance(self._total_reward_loss, nn.Variable)

        self._binary_regression_loss.persistent = True
        self._grad_penalty_loss.persistent = True
        self._regularization_loss.persistent = True
        self._extra_regularization_loss.persistent = True
        self._total_reward_loss.persistent = True

    def _build_one_step_graph(
        self,
        models: Sequence[Model],
        training_variables: TrainingVariables,
        ignore_loss: bool,
    ):
        if ignore_loss:
            return

        models = cast(Sequence[RewardFunction], models)
        for model in models:
            binary_regression_loss, grad_penalty_loss = self._build_adversarial_loss(model, training_variables)
            self._binary_regression_loss += binary_regression_loss
            self._grad_penalty_loss += grad_penalty_loss

            self._regularization_loss += self._build_regularization_penalty(model)

            self._extra_regularization_loss += self._build_extra_regularization_penalty(model)

    def _setup_training_variables(self, batch_size):
        s_current_agent_var = create_variable(batch_size, self._env_info.state_shape)
        s_next_agent_var = create_variable(batch_size, self._env_info.state_shape)

        s_current_expert_var = create_variable(batch_size, self._env_info.state_shape)
        s_next_expert_var = create_variable(batch_size, self._env_info.state_shape)

        a_current_agent_var = create_variable(batch_size, self._env_info.action_shape)
        a_current_expert_var = create_variable(batch_size, self._env_info.action_shape)

        variables = {
            "s_current_expert": s_current_expert_var,
            "a_current_expert": a_current_expert_var,
            "s_next_expert": s_next_expert_var,
            "s_current_agent": s_current_agent_var,
            "a_current_agent": a_current_agent_var,
            "s_next_agent": s_next_agent_var,
        }

        training_variables = TrainingVariables(batch_size, extra=variables)

        return training_variables

    def _build_adversarial_loss(self, model: RewardFunction, training_variables: TrainingVariables):
        s_expert, s_n_expert, s_agent, s_n_agent = self._preprocess_state(training_variables)

        _apply_need_grad_true(s_expert)
        _apply_need_grad_true(s_n_expert)

        logits_real, logits_fake = self._compute_logits(
            model,
            s_expert,
            training_variables.extra["a_current_expert"],
            s_n_expert,
            s_agent,
            training_variables.extra["a_current_agent"],
            s_n_agent,
        )
        real_loss = 0.5 * NF.mean((logits_real - 1.0) ** 2)
        fake_loss = 0.5 * NF.mean((logits_fake + 1.0) ** 2)
        binary_regression_loss = 0.5 * (real_loss + fake_loss)

        # grad penalty for expert state
        current_state_grads = self._compute_gradient_wrt_state(logits_real, s_expert)
        current_state_grad_penalty = self._compute_gradient_penalty(current_state_grads)

        next_state_grads = self._compute_gradient_wrt_state(logits_real, s_n_expert)
        next_state_grad_penalty = self._compute_gradient_penalty(next_state_grads)

        grad_penalty_loss = self._config.gradient_penelty_coefficient * (
            current_state_grad_penalty + next_state_grad_penalty
        )

        return binary_regression_loss, grad_penalty_loss

    def _preprocess_state(self, training_variables: TrainingVariables):
        s_expert = training_variables.extra["s_current_expert"]
        s_n_expert = training_variables.extra["s_next_expert"]
        if self._state_preprocessor is not None:
            s_expert = self._state_preprocessor.process(s_expert)
            s_n_expert = self._state_preprocessor.process(s_n_expert)

        s_agent = training_variables.extra["s_current_agent"]
        s_n_agent = training_variables.extra["s_next_agent"]
        if self._state_preprocessor is not None:
            s_agent = self._state_preprocessor.process(s_agent)
            s_n_agent = self._state_preprocessor.process(s_n_agent)

        return s_expert, s_n_expert, s_agent, s_n_agent

    def _compute_logits(self, model, s_expert, a_expert, s_n_expert, s_agent, a_agent, s_n_agent):
        logits_real = model.r(s_expert, a_expert, s_n_expert)
        assert logits_real.shape[1] == 1

        logits_fake = model.r(s_agent, a_agent, s_n_agent)
        assert logits_fake.shape[1] == 1

        return logits_real, logits_fake

    def _build_regularization_penalty(self, model: RewardFunction):
        regularization_loss = 0.0
        model_params = model.get_parameters()
        for n, p in model_params.items():
            # without bias
            if not ("/b" in n):
                regularization_loss += self._config.regularization_coefficient * 0.5 * NF.sum(p**2)
        return regularization_loss

    def _build_extra_regularization_penalty(self, model: RewardFunction):
        extra_regularization_loss = 0.0
        model_params = model.get_parameters()
        for variable_name in self._config.extra_regularization_variable_names:
            extra_regularization_loss += (
                self._config.extra_regularization_coefficient * 0.5 * NF.sum(model_params[variable_name] ** 2)
            )
        return extra_regularization_loss

    def _compute_gradient_wrt_state(
        self, loss: nn.Variable, state_variable: Union[nn.Variable, Tuple[nn.Variable, ...]]
    ) -> List[nn.Variable]:
        if isinstance(state_variable, nn.Variable) or isinstance(state_variable, tuple):
            grads: List[nn.Variable] = nn.grad(loss, state_variable)
        else:
            raise ValueError

        for g in grads:
            g.persistent = True

        return grads

    def _compute_gradient_penalty(self, grads: List[nn.Variable]) -> nn.Variable:
        valid_grads: List[nn.Variable] = []
        indexes = (
            range(len(grads))
            if self._config.gradient_penalty_indexes is None
            else self._config.gradient_penalty_indexes
        )
        for i in indexes:
            g = grads[i]
            # if gradient is empty, skip to take sum.
            if len(g.shape) == 0:
                continue
            valid_grads.append(g)

        if len(valid_grads) == 0:
            return 0.0
        elif len(valid_grads) == 1:
            return 0.5 * NF.mean(NF.sum(valid_grads[0] ** 2, axis=-1))
        else:
            concat_g: nn.Variable = NF.concatenate(*valid_grads, axis=-1)
            return 0.5 * NF.mean(NF.sum(concat_g**2, axis=-1))

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"reward_loss": self._total_reward_loss}


def _apply_need_grad_true(variable: Union[nn.Variable, Tuple[nn.Variable, ...]]):
    if isinstance(variable, tuple):
        for v in variable:
            v.need_grad = True
    else:
        variable.need_grad = True
