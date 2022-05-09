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
from typing import Dict, Iterable, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables)
from nnabla_rl.models import Model, RewardFunction
from nnabla_rl.utils.data import convert_to_list_if_not_list, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class GAILRewardFunctionTrainerConfig(TrainerConfig):
    batch_size: int = 1024
    learning_rate: float = 3e-4
    entropy_coef: float = 0.001

    def __post_init__(self):
        self._assert_positive(self.entropy_coef, "entropy_coef")


class GAILRewardFunctionTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: GAILRewardFunctionTrainerConfig
    _binary_classification_loss: nn.Variable

    def __init__(self,
                 models: Union[RewardFunction, Sequence[RewardFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info,
                 config=GAILRewardFunctionTrainerConfig()):
        super(GAILRewardFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            for key in batch.extra.keys():
                set_data_to_variable(t.extra[key], b.extra[key])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._binary_classification_loss.forward()
        self._binary_classification_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state: Dict[str, np.ndarray] = {}
        trainer_state['reward_loss'] = self._binary_classification_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Union[Model, Sequence[Model]],
                              training_variables: TrainingVariables):
        models = convert_to_list_if_not_list(models)
        models = cast(Sequence[RewardFunction], models)

        self._binary_classification_loss = 0
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
        models = cast(Sequence[RewardFunction], models)
        for model in models:
            # fake path
            logits_fake = model.r(training_variables.extra['s_current_agent'],
                                  training_variables.extra['a_current_agent'],
                                  training_variables.extra['s_next_agent'])
            fake_loss = NF.mean(NF.sigmoid_cross_entropy(logits_fake, NF.constant(0, logits_fake.shape)))
            # real path
            logits_real = model.r(training_variables.extra['s_current_expert'],
                                  training_variables.extra['a_current_expert'],
                                  training_variables.extra['s_next_expert'])
            real_loss = NF.mean(NF.sigmoid_cross_entropy(logits_real, NF.constant(1, logits_real.shape)))
            # entropy loss
            logits = NF.concatenate(logits_fake, logits_real, axis=0)
            entropy = NF.mean((1. - NF.sigmoid(logits)) * logits - NF.log_sigmoid(logits))
            entropy_loss = - self._config.entropy_coef * entropy  # maximize
            self._binary_classification_loss += 0.0 if ignore_loss else fake_loss + real_loss + entropy_loss

    def _setup_training_variables(self, batch_size):
        s_current_agent_var = create_variable(batch_size, self._env_info.state_shape)
        s_next_agent_var = create_variable(batch_size, self._env_info.state_shape)
        s_current_expert_var = create_variable(batch_size, self._env_info.state_shape)
        s_next_expert_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_agent_var = create_variable(batch_size, self._env_info.action_shape)
        a_current_expert_var = create_variable(batch_size, self._env_info.action_shape)

        variables = {'s_current_expert': s_current_expert_var,
                     'a_current_expert': a_current_expert_var,
                     's_next_expert': s_next_expert_var,
                     's_current_agent': s_current_agent_var,
                     'a_current_agent': a_current_agent_var,
                     's_next_agent': s_next_agent_var}

        return TrainingVariables(batch_size, extra=variables)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"binary_classification_loss": self._binary_classification_loss}
