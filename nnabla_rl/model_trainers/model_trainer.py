# Copyright 2020,2021 Sony Corporation.
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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np

import nnabla as nn
from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import Model
from nnabla_rl.utils.data import convert_to_list_if_not_list


@dataclass
class TrainerConfig(Configuration):
    """Configuration class for ModelTrainer
    """

    def __post_init__(self):
        super(TrainerConfig, self).__post_init__()


class TrainingBatch():
    """Mini-Batch class for train

    Args:
       batch_size (int): the size of mini-batch
       s_current (Optional[np.array]): the current state array
       a_current (Optional[np.array]): the current action array
       reward (Optional[np.array]): the reward value array
       gamma (Optional[float]): gamma value
       non_terminal (Optional[np.array]): the non_terminal flag array
       s_next (Optional[np.array]): the next state array
       weight (Optional[np.array]): the weight of loss array
       extra (Dict[str, np.array]): the extra information
       next_step_batch (Optional[:py:class:`TrainingBatch <nnabla_rl.model_trainers.model_trainer.TrainingBatch>`]):\
           the mini-batch for next step (used in n-step learning)
    """
    batch_size: int
    s_current: np.array
    a_current: np.array
    reward: np.array
    gamma: float
    non_terminal: np.array
    s_next: np.array
    weight: np.array
    extra: Dict[str, np.array]
    # Used in n-step learning
    next_step_batch: Optional['TrainingBatch']

    def __init__(self,
                 batch_size: int,
                 s_current: Optional[np.array] = None,
                 a_current: Optional[np.array] = None,
                 reward: Optional[np.array] = None,
                 gamma: Optional[float] = None,
                 non_terminal: Optional[np.array] = None,
                 s_next: Optional[np.array] = None,
                 weight: Optional[np.array] = None,
                 extra: Dict[str, np.array] = {},
                 next_step_batch: Optional['TrainingBatch'] = None):
        assert 0 < batch_size
        self.batch_size = batch_size
        if s_current is not None:
            self.s_current = s_current
        if a_current is not None:
            self.a_current = a_current
        if reward is not None:
            self.reward = reward
        if gamma is not None:
            self.gamma = gamma
        if non_terminal is not None:
            self.non_terminal = non_terminal
        if s_next is not None:
            self.s_next = s_next
        if weight is not None:
            self.weight = weight
        self.extra: Dict[str, np.array] = extra
        self.next_step_batch = next_step_batch


class TrainingVariables():
    batch_size: int
    s_current: nn.Variable
    a_current: nn.Variable
    reward: nn.Variable
    gamma: nn.Variable
    non_terminal: nn.Variable
    s_next: nn.Variable
    weight: nn.Variable

    def __init__(self,
                 batch_size: int,
                 s_current: Optional[nn.Variable] = None,
                 a_current: Optional[nn.Variable] = None,
                 reward: Optional[nn.Variable] = None,
                 gamma: Optional[nn.Variable] = None,
                 non_terminal: Optional[nn.Variable] = None,
                 s_next: Optional[nn.Variable] = None,
                 weight: Optional[nn.Variable] = None,
                 extra: Dict[str, nn.Variable] = {}):
        assert 0 < batch_size
        self.batch_size = batch_size
        if s_current is not None:
            self.s_current = s_current
        if a_current is not None:
            self.a_current = a_current
        if reward is not None:
            self.reward = reward
        if gamma is not None:
            self.gamma = gamma
        if non_terminal is not None:
            self.non_terminal = non_terminal
        if s_next is not None:
            self.s_next = s_next
        if weight is not None:
            self.weight = weight
        self.extra: Dict[str, nn.Variable] = extra


class ModelTrainer(metaclass=ABCMeta):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _config: TrainerConfig
    _models: Sequence[Model]
    _solvers: Dict[str, nn.solver.Solver]
    _train_count: int
    _training_variables: TrainingVariables

    def __init__(self,
                 models: Union[Model, Sequence[Model]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: TrainerConfig):
        self._env_info = env_info
        self._config = config

        self._train_count = 0

        self._models = convert_to_list_if_not_list(models)
        self._solvers = solvers

        # Initially create training variables with batch_size 1.
        # The batch_size will be updated later depending on the given experience data
        # This procedure is a workaround to initialize model parameters (it it is not created).
        self._training_variables = self._setup_training_variables(1)

        self._build_training_graph(self._models, self._training_variables)

        self._setup_solver()

    def train(self, batch: TrainingBatch, **kwargs) -> Dict[str, np.array]:
        if self._models is None:
            raise RuntimeError('Call setup_training() first. Model is not set!')
        self._train_count += 1

        batch = self._setup_batch(batch)
        new_batch_size = batch.batch_size
        prev_batch_size = self._training_variables.batch_size
        if new_batch_size != prev_batch_size:
            self._training_variables = self._setup_training_variables(new_batch_size)
            self._build_training_graph(self._models, self._training_variables)

        trainer_state = self._update_model(self._models, self._solvers, batch, self._training_variables, **kwargs)

        return trainer_state

    def set_learning_rate(self, new_learning_rate):
        for solver in self._solvers.values():
            solver.set_learning_rate(new_learning_rate)

    def _setup_batch(self, batch: TrainingBatch) -> TrainingBatch:
        return batch

    @abstractmethod
    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        raise NotImplementedError

    @abstractmethod
    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        raise NotImplementedError

    @abstractmethod
    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        raise NotImplementedError

    def _setup_solver(self):
        for model in self._models:
            if model.scope_name in self._solvers.keys():
                solver = self._solvers[model.scope_name]
                # Set retain_state = True and prevent overwriting loaded state (If it is loaded)
                solver.set_parameters(model.get_parameters(), reset=False, retain_state=True)
