# Copyright 2021 Sony Group Corporation.
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
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, StochasticPolicy


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
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current
        training_variables.extra['log_prob'].d = batch.extra['log_prob']
        training_variables.extra['advantage'].d = batch.extra['advantage']
        training_variables.extra['alpha'].d = batch.extra['alpha']

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = float(self._pi_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)
        self._pi_loss = 0
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

            self._pi_loss += -clip_loss - self._config.entropy_coefficient * entropy_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            action_shape = (batch_size, 1)
        else:
            action_shape = (batch_size, self._env_info.action_dim)
        a_current_var = nn.Variable(action_shape)
        log_prob_var = nn.Variable((batch_size, 1))
        advantage_var = nn.Variable((batch_size, 1))
        alpha_var = nn.Variable((batch_size, 1))

        extra = {}
        extra['log_prob'] = log_prob_var
        extra['advantage'] = advantage_var
        extra['alpha'] = alpha_var
        return TrainingVariables(batch_size, s_current_var, a_current_var, extra=extra)
