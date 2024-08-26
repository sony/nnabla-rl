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
from typing import Dict, Sequence, Union, cast

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
from nnabla_rl.models import Model, OptionValueFunction, StochasticTerminationFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class OptionCriticTerminationFunctionTrainerConfig(TrainerConfig):
    advantage_offset: float = 0.01
    reduction_method: str = "sum"

    def __post_init__(self):
        self._assert_positive_or_zero(self.advantage_offset, "advantage_offset")


class OptionCriticTerminationFunctionTrainer(ModelTrainer):
    """Option Critic Termination function Trainer."""

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: OptionCriticTerminationFunctionTrainerConfig
    _termination_loss: nn.Variable

    def __init__(
        self,
        models: Union[StochasticTerminationFunction, Sequence[StochasticTerminationFunction]],
        solvers: Dict[str, nn.solver.Solver],
        env_info: EnvironmentInfo,
        option_v_functions: Union[OptionValueFunction, Sequence[OptionValueFunction]],
        config: OptionCriticTerminationFunctionTrainerConfig = OptionCriticTerminationFunctionTrainerConfig(),
    ):
        self._option_v_functions = convert_to_list_if_not_list(option_v_functions)
        super(OptionCriticTerminationFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(
        self,
        models: Sequence[Model],
        solvers: Dict[str, nn.solver.Solver],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_next, b.s_next)
            set_data_to_variable(t.extra["option"], b.extra["option"])
            set_data_to_variable(t.non_terminal, b.non_terminal)

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._termination_loss.forward()
        self._termination_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state["termination_loss"] = self._termination_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticTerminationFunction], models)

        self._termination_loss = 0.0

        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        # NOTE: In author's implementation they use s_current in s_next.
        # However, we follow the author's paper instead.
        models = cast(Sequence[StochasticTerminationFunction], models)

        # compute advantage
        advantage = 0.0
        for option_v_function in self._option_v_functions:
            max_option_v = option_v_function.max_option_v(training_variables.s_next)
            option_v = option_v_function.option_v(training_variables.s_next, training_variables.extra["option"])
            advantage += option_v - max_option_v + self._config.advantage_offset

        # NOTE: use mean in default
        advantage = advantage / float(len(self._option_v_functions))
        assert isinstance(advantage, nn.Variable)
        advantage.need_grad = False

        for termination_function in models:
            distribution = termination_function.termination(
                training_variables.s_next, training_variables.extra["option"]
            )
            if self._config.reduction_method == "sum":
                loss = NF.sum(distribution.mean() * advantage)
            elif self._config.reduction_method == "mean":
                loss = NF.mean(distribution.mean() * advantage)
            else:
                raise ValueError

            self._termination_loss += 0.0 if ignore_loss else loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        s_next_var = create_variable(batch_size, self._env_info.state_shape)
        option_var = create_variable(batch_size, 1)
        non_terminal_var = create_variable(batch_size, 1)

        extra = {}
        extra["option"] = option_var
        return TrainingVariables(batch_size=batch_size, s_next=s_next_var, extra=extra, non_terminal=non_terminal_var)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"termination_loss": self._termination_loss}
