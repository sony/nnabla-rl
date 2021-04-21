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
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch
from nnabla_rl.models import Model


@dataclass
class MultiStepTrainerConfig(TrainerConfig):
    """Configuration class for ModelTrainer
    """
    num_steps: int = 1

    def __post_init__(self):
        super(MultiStepTrainerConfig, self).__post_init__()
        self._assert_positive(self.num_steps, 'num_steps')


class MultiStepTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: MultiStepTrainerConfig

    def __init__(self,
                 models: Union[Model, Sequence[Model]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: TrainerConfig):
        super(MultiStepTrainer, self).__init__(models, solvers, env_info, config)

    def _setup_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        if self._config.num_steps == 1:
            return training_batch
        else:
            n_step_non_terminal = np.copy(training_batch.non_terminal)
            n_step_reward = np.copy(training_batch.reward)
            n_step_gamma = np.copy(training_batch.gamma)
            n_step_state = np.copy(training_batch.s_next)
            next_batch = cast(TrainingBatch, training_batch.next_step_batch)

            for _ in range(self._config.num_steps - 1):
                # Do not add reward if previous state is terminal state
                n_step_reward += next_batch.reward * n_step_non_terminal * n_step_gamma
                n_step_non_terminal *= next_batch.non_terminal
                n_step_gamma *= next_batch.gamma
                n_step_state = next_batch.s_next

                next_batch = cast(TrainingBatch, next_batch.next_step_batch)

            return TrainingBatch(batch_size=training_batch.batch_size,
                                 s_current=training_batch.s_current,
                                 a_current=training_batch.a_current,
                                 reward=n_step_reward,
                                 gamma=n_step_gamma,
                                 non_terminal=n_step_non_terminal,
                                 s_next=n_step_state,
                                 weight=training_batch.weight,
                                 extra=training_batch.extra,
                                 next_step_batch=None)
