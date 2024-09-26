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
from typing import Dict, Optional, Sequence, Union

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (
    LossIntegration,
    ModelTrainer,
    TrainerConfig,
    TrainingBatch,
    TrainingVariables,
)
from nnabla_rl.models import Model, QFunction, StochasticPolicy, VFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class AWRPolicyTrainerConfig(TrainerConfig):
    """List of configuration for AWRPolicyTrainer.

    Args:
        beta (float): the temperature parameter of advantage weight. Defaults to 1.0
        advantage_clip (Optional[float]): the value for clipping advantage weight. Defaults to 100.0
    """

    beta: float = 1.0
    advantage_clip: Optional[float] = 100.0


class AWRPolicyTrainer(ModelTrainer):
    """Policy Trainer with Advantage-Weighted Regression (AWR)"""

    _pi_loss: nn.Variable
    _config: AWRPolicyTrainerConfig

    def __init__(
        self,
        models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
        solvers: Dict[str, nn.solver.Solver],
        q_functions: Union[QFunction, Sequence[QFunction]],
        v_function: VFunction,
        env_info: EnvironmentInfo,
        config: AWRPolicyTrainerConfig = AWRPolicyTrainerConfig(),
    ):
        self._q_functions = convert_to_list_if_not_list(q_functions)
        self._v_function = v_function
        super(AWRPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        self._pi_loss = 0
        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._pi_loss += self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        advantage_weight = self._compute_advantage_weight(training_variables)
        advantage_weight.need_grad = False
        one_step_pi_loss = 0.0
        for policy in models:
            assert isinstance(policy, StochasticPolicy)

            # Actor optimization graph
            policy_distribution = policy.pi(training_variables.s_current)
            log_pi = policy_distribution.log_prob(training_variables.a_current)
            one_step_pi_loss += 0.0 if ignore_loss else NF.mean(advantage_weight * log_pi) * (-1)
        return one_step_pi_loss

    def _compute_advantage_weight(self, training_variables: TrainingVariables):
        q_values = []
        for q_function in self._q_functions:
            q_values.append(q_function.q(training_variables.s_current, training_variables.a_current))
        min_q = RNF.minimum_n(q_values)
        v = self._v_function.v(training_variables.s_current)
        advantage = min_q - v
        advantage_weight = NF.exp(self._config.beta * advantage)
        if self._config.advantage_clip is not None:
            advantage_weight = NF.minimum_scalar(advantage_weight, val=self._config.advantage_clip)
        return advantage_weight

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

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward()
        self._pi_loss.backward()
        for solver in solvers.values():
            solver.update()
        trainer_state = {}
        trainer_state["pi_loss"] = self._pi_loss.d.copy()
        return trainer_state

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        return TrainingVariables(batch_size, s_current_var, a_current_var)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}
