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
                                                    TrainingVariables)
from nnabla_rl.models import Model, StochasticPolicy
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class PPOPolicyTrainerConfig(TrainerConfig):
    entropy_coefficient: float = 0.01
    epsilon: float = 0.1


class PPOPolicyTrainer(ModelTrainer):
    '''Proximal Policy Optimization (PPO) style Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: PPOPolicyTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: PPOPolicyTrainerConfig = PPOPolicyTrainerConfig()):
        super(PPOPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.extra['log_prob'], b.extra['log_prob'])
            set_data_to_variable(t.extra['advantage'], b.extra['advantage'])
            set_data_to_variable(t.extra['alpha'], b.extra['alpha'])

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
        models = cast(Sequence[StochasticPolicy], models)
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
        models = cast(Sequence[StochasticPolicy], models)
        for policy in models:
            distribution = policy.pi(training_variables.s_current)
            log_prob_new = distribution.log_prob(training_variables.a_current)
            log_prob_old = training_variables.extra['log_prob']
            probability_ratio = NF.exp(log_prob_new - log_prob_old)
            alpha = training_variables.extra['alpha']
            clipped_ratio = NF.clip_by_value(probability_ratio,
                                             1 - self._config.epsilon * alpha,
                                             1 + self._config.epsilon * alpha)
            advantage = training_variables.extra['advantage']
            lower_bounds = NF.minimum2(probability_ratio * advantage, clipped_ratio * advantage)
            clip_loss = NF.mean(lower_bounds)

            entropy = distribution.entropy()
            entropy_loss = NF.mean(entropy)

            self._pi_loss += 0.0 if ignore_loss else (-clip_loss - self._config.entropy_coefficient * entropy_loss)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        log_prob_var = create_variable(batch_size, 1)
        advantage_var = create_variable(batch_size, 1)
        alpha_var = create_variable(batch_size, 1)

        extra = {}
        extra['log_prob'] = log_prob_var
        extra['advantage'] = advantage_var
        extra['alpha'] = alpha_var
        return TrainingVariables(batch_size, s_current_var, a_current_var, extra=extra)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}
