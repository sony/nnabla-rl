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
from typing import Dict, Optional, Sequence, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, StochasticPolicy
from nnabla_rl.utils.misc import clip_grad_by_global_norm


@dataclass
class A2CPolicyTrainerConfig(TrainerConfig):
    entropy_coefficient: float = 0.01
    max_grad_norm: Optional[float] = 0.5


class A2CPolicyTrainer(ModelTrainer):
    '''Advantaged Actor Critic style Policy Trainer
    '''
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: A2CPolicyTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: A2CPolicyTrainerConfig = A2CPolicyTrainerConfig()):
        super(A2CPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current
        training_variables.extra['advantage'].d = batch.extra['advantage']

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward()
        self._pi_loss.backward()
        for solver in solvers.values():
            if self._config.max_grad_norm is not None:
                clip_grad_by_global_norm(solver, self._config.max_grad_norm)
            solver.update()

        trainer_state = {}
        trainer_state['pi_loss'] = float(self._pi_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)
        self._pi_loss = 0
        for policy in models:
            distribution = policy.pi(training_variables.s_current)
            log_prob = distribution.log_prob(training_variables.a_current)
            entropy = distribution.entropy()
            advantage = training_variables.extra['advantage']

            self._pi_loss += NF.mean(-advantage * log_prob - self._config.entropy_coefficient * entropy)

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((batch_size, 1))
        else:
            a_current_var = nn.Variable((batch_size, self._env_info.action_dim))
        advantage_var = nn.Variable((batch_size, 1))
        extra = {}
        extra['advantage'] = advantage_var
        return TrainingVariables(batch_size, s_current_var, a_current_var, extra=extra)
