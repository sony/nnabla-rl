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
from nnabla_rl.models import DiscreteOptionValueFunction, Model, StochasticTerminationFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class OptionCriticOptionValueTrainerConfig(TrainerConfig):
    reduction_method: str = "sum"


class OptionCriticOptionValueTrainer(ModelTrainer):
    """Option Critic Option Value Trainer."""

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: OptionCriticOptionValueTrainerConfig
    _option_v_loss: nn.Variable

    def __init__(
        self,
        train_functions: Union[DiscreteOptionValueFunction, Sequence[DiscreteOptionValueFunction]],
        target_function: DiscreteOptionValueFunction,
        solvers: Dict[str, nn.solver.Solver],
        env_info: EnvironmentInfo,
        termination_functions: Union[StochasticTerminationFunction, Sequence[StochasticTerminationFunction]],
        config: OptionCriticOptionValueTrainerConfig = OptionCriticOptionValueTrainerConfig(),
    ):
        self._target_function = target_function
        self._termination_functions = convert_to_list_if_not_list(termination_functions)
        super(OptionCriticOptionValueTrainer, self).__init__(train_functions, solvers, env_info, config)

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
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.reward, b.reward)
            set_data_to_variable(t.gamma, b.gamma)
            set_data_to_variable(t.non_terminal, b.non_terminal)
            set_data_to_variable(t.s_next, b.s_next)
            set_data_to_variable(t.extra["option"], b.extra["option"])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._option_v_loss.forward()
        self._option_v_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state["option_v_loss"] = self._option_v_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DiscreteOptionValueFunction], models)

        self._option_v_loss = 0.0

        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        models = cast(Sequence[DiscreteOptionValueFunction], models)

        next_s_curr_option_termination_prob = 0.0
        for termination_function in self._termination_functions:
            termination_dist = termination_function.termination(
                training_variables.s_next, training_variables.extra["option"]
            )
            distribution_mean = termination_dist.mean()
            assert isinstance(distribution_mean, nn.Variable)
            next_s_curr_option_termination_prob += distribution_mean

        # NOTE: use mean in default
        next_s_curr_option_termination_prob = next_s_curr_option_termination_prob / float(
            len(self._termination_functions)
        )
        assert isinstance(next_s_curr_option_termination_prob, nn.Variable)
        next_s_curr_option_termination_prob.need_grad = False

        # current_option_q.shape = (batch_size, 1)
        next_s_curr_option_v = self._target_function.option_v(
            training_variables.s_next, training_variables.extra["option"]
        )
        # max_option_q_in_next.shape = (batch_size, 1)
        next_s_max_option_v = self._target_function.max_option_v(training_variables.s_next)

        target = training_variables.reward + training_variables.non_terminal * training_variables.gamma * (
            (1.0 - next_s_curr_option_termination_prob) * next_s_curr_option_v
            + next_s_curr_option_termination_prob * next_s_max_option_v
        )
        target.need_grad = False

        for option_v_function in models:
            option_v = option_v_function.option_v(training_variables.s_current, training_variables.extra["option"])

            if self._config.reduction_method == "sum":
                loss = 0.5 * NF.sum(NF.huber_loss(target, option_v))
            elif self._config.reduction_method == "mean":
                loss = 0.5 * NF.mean(NF.huber_loss(target, option_v))
            else:
                raise ValueError

            self._option_v_loss += 0.0 if ignore_loss else loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        s_next_var = create_variable(batch_size, self._env_info.state_shape)
        reward_var = create_variable(batch_size, 1)
        gamma_var = create_variable(batch_size, 1)
        non_terminal_var = create_variable(batch_size, 1)
        option_var = create_variable(batch_size, 1)

        extra = {}
        extra["option"] = option_var

        training_variables = TrainingVariables(
            batch_size=batch_size,
            s_current=s_current_var,
            a_current=a_current_var,
            reward=reward_var,
            gamma=gamma_var,
            non_terminal=non_terminal_var,
            s_next=s_next_var,
            extra=extra,
        )

        return training_variables

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"option_v_loss": self._option_v_loss}
