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
                 config: MultiStepTrainerConfig):
        super(MultiStepTrainer, self).__init__(models, solvers, env_info, config)

    def _setup_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        if self._config.num_steps == 1:
            return training_batch
        else:
            training_batch_length = len(training_batch)
            total_timesteps = self._config.unroll_steps + self._config.burn_in_steps
            assert training_batch_length == (total_timesteps + self._config.num_steps - 1)
            n_step_reward = np.zeros(shape=(training_batch.batch_size, self._config.num_steps))
            n_step_gamma = np.ones(shape=(training_batch.batch_size, self._config.num_steps))
            n_step_non_terminal = np.ones(shape=(training_batch.batch_size, self._config.num_steps))
            n_step_batch = None
            next_step_batch = None

            training_batch_list = list(training_batch)
            for i, batch in enumerate(reversed(training_batch_list)):
                n_step_reward = np.roll(n_step_reward, 1)
                n_step_gamma = np.roll(n_step_gamma, 1)
                n_step_non_terminal = np.roll(n_step_non_terminal, 1)

                n_step_reward[:, 0:1] = batch.reward
                n_step_reward[:, 1:] *= batch.non_terminal * batch.gamma
                n_step_gamma[:, 0:1] = batch.gamma
                n_step_non_terminal[:, 0:1] = batch.non_terminal

                if i < self._config.num_steps - 1:
                    continue

                last_batch = training_batch_list[training_batch_length - 1 - (i - self._config.num_steps + 1)]
                n_step_batch = TrainingBatch(batch_size=batch.batch_size,
                                             s_current=batch.s_current,
                                             a_current=batch.a_current,
                                             reward=np.sum(n_step_reward, axis=1, keepdims=True),
                                             gamma=np.prod(n_step_gamma, axis=1, keepdims=True),
                                             non_terminal=np.prod(n_step_non_terminal, axis=1, keepdims=True),
                                             s_next=last_batch.s_next,
                                             weight=batch.weight,
                                             extra=batch.extra,
                                             next_step_batch=next_step_batch)
                next_step_batch = n_step_batch

            return cast(TrainingBatch, n_step_batch)
