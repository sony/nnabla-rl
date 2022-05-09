# Copyright 2020,2021 Sony Corporation.
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

import contextlib
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

import nnabla as nn
from nnabla_rl.configuration import Configuration
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import Model
from nnabla_rl.utils.data import convert_to_list_if_not_list
from nnabla_rl.utils.misc import retrieve_internal_states


@contextlib.contextmanager
def rnn_support(model: Model,
                prev_rnn_states: Dict[str, Dict[str, nn.Variable]],
                train_rnn_states: Dict[str, Dict[str, nn.Variable]],
                training_variables: 'TrainingVariables',
                config: 'TrainerConfig'):
    def stop_backprop(rnn_states):
        for value in rnn_states.values():
            value.need_grad = False
    try:
        if model.is_recurrent():
            scope_name = model.scope_name
            internal_states = retrieve_internal_states(
                scope_name, prev_rnn_states, train_rnn_states, training_variables, config.reset_on_terminal)
            model.set_internal_states(internal_states)
        yield
    finally:
        if model.is_recurrent():
            rnn_states = model.get_internal_states()
            if training_variables.step_index() < config.burn_in_steps:
                stop_backprop(rnn_states)
            prev_rnn_states[model.scope_name] = rnn_states


class LossIntegration(Enum):
    ALL_TIMESTEPS = 1, 'Computed loss is summed over all timesteps'
    LAST_TIMESTEP_ONLY = 2, 'Only the last timestep\'s loss is used.'


@dataclass
class TrainerConfig(Configuration):
    """Configuration class for ModelTrainer
    """
    unroll_steps: int = 1
    burn_in_steps: int = 0
    reset_on_terminal: bool = True  # Reset internal rnn state to given state if previous state is terminal.
    loss_integration: LossIntegration = LossIntegration.ALL_TIMESTEPS

    def __post_init__(self):
        super(TrainerConfig, self).__post_init__()
        self._assert_positive(self.unroll_steps, 'unroll_steps')
        self._assert_positive_or_zero(self.burn_in_steps, 'burn_in_steps')


class TrainingBatch():
    """Mini-Batch class for train

    Args:
       batch_size (int): the size of mini-batch
       s_current (Optional[np.ndarray]): the current state array
       a_current (Optional[np.ndarray]): the current action array
       reward (Optional[np.ndarray]): the reward value array
       gamma (Optional[float]): gamma value
       non_terminal (Optional[np.ndarray]): the non_terminal flag array
       s_next (Optional[np.ndarray]): the next state array
       weight (Optional[np.ndarray]): the weight of loss array
       extra (Dict[str, np.ndarray]): the extra information
       next_step_batch (Optional[:py:class:`TrainingBatch <nnabla_rl.model_trainers.model_trainer.TrainingBatch>`]):\
           the mini-batch for next step (used in n-step learning)
       rnn_states (Dict[str, Dict[str, np.array]]): the rnn internal state values
    """
    batch_size: int
    s_current: Union[np.ndarray, Tuple[np.ndarray, ...]]
    a_current: np.ndarray
    reward: np.ndarray
    gamma: float
    non_terminal: np.ndarray
    s_next: Union[np.ndarray, Tuple[np.ndarray, ...]]
    weight: np.ndarray
    extra: Dict[str, np.ndarray]
    # Used in n-step/rnn learning
    next_step_batch: Optional['TrainingBatch']
    rnn_states: Dict[str, Dict[str, np.ndarray]]

    def __init__(self,
                 batch_size: int,
                 s_current: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None,
                 a_current: Optional[np.ndarray] = None,
                 reward: Optional[np.ndarray] = None,
                 gamma: Optional[float] = None,
                 non_terminal: Optional[np.ndarray] = None,
                 s_next: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None,
                 weight: Optional[np.ndarray] = None,
                 extra: Dict[str, np.ndarray] = {},
                 next_step_batch: Optional['TrainingBatch'] = None,
                 rnn_states: Dict[str, Dict[str, np.ndarray]] = {}):
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
        self.extra: Dict[str, np.ndarray] = extra
        self.next_step_batch = next_step_batch
        self.rnn_states = rnn_states

    def __getitem__(self, index):
        num_steps = len(self)
        if num_steps <= index:
            raise IndexError

        batch = self
        for _ in range(index):
            batch = batch.next_step_batch
        return batch

    def __iter__(self):
        batch = self
        while batch is not None:
            yield batch
            batch = batch.next_step_batch

    def __len__(self):
        num_steps = 1
        batch = self.next_step_batch
        while batch is not None:
            num_steps += 1
            batch = batch.next_step_batch
        return num_steps


class TrainingVariables():
    batch_size: int
    s_current: Union[nn.Variable, Tuple[nn.Variable, ...]]
    a_current: nn.Variable
    reward: nn.Variable
    gamma: nn.Variable
    non_terminal: nn.Variable
    s_next: Union[nn.Variable, Tuple[nn.Variable, ...]]
    weight: nn.Variable
    extra: Dict[str, nn.Variable]
    rnn_states: Dict[str, Dict[str, nn.Variable]]

    # Used in rnn learning
    _next_step_variables: Optional['TrainingVariables']
    _prev_step_variables: Optional['TrainingVariables']

    def __init__(self,
                 batch_size: int,
                 s_current: Optional[Union[nn.Variable, Tuple[nn.Variable, ...]]] = None,
                 a_current: Optional[nn.Variable] = None,
                 reward: Optional[nn.Variable] = None,
                 gamma: Optional[nn.Variable] = None,
                 non_terminal: Optional[nn.Variable] = None,
                 s_next: Optional[Union[nn.Variable, Tuple[nn.Variable, ...]]] = None,
                 weight: Optional[nn.Variable] = None,
                 extra: Dict[str, nn.Variable] = {},
                 next_step_variables: Optional["TrainingVariables"] = None,
                 rnn_states: Dict[str, Dict[str, nn.Variable]] = {}):
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
        self.next_step_variables = next_step_variables
        self._prev_step_variables = None
        self.rnn_states = rnn_states

    @property
    def next_step_variables(self) -> Optional["TrainingVariables"]:
        return self._next_step_variables

    @next_step_variables.setter
    def next_step_variables(self, value: Optional["TrainingVariables"]) -> None:
        self._next_step_variables = value
        if self._next_step_variables is None:
            return
        if self._next_step_variables.prev_step_variables is not self:
            self._next_step_variables.prev_step_variables = self

    @property
    def prev_step_variables(self) -> Optional["TrainingVariables"]:
        return self._prev_step_variables

    @prev_step_variables.setter
    def prev_step_variables(self, value: Optional["TrainingVariables"]) -> None:
        self._prev_step_variables = value
        if self._prev_step_variables is None:
            return
        if self._prev_step_variables.next_step_variables is not self:
            self._prev_step_variables.next_step_variables = self

    def __getitem__(self, item: int) -> "TrainingVariables":
        num_steps = len(self)
        if num_steps <= item:
            raise IndexError

        variable = self
        for _ in range(item):
            assert variable.next_step_variables
            variable = variable.next_step_variables
        assert variable
        return variable

    def __iter__(self):
        variable = self
        while variable is not None:
            yield variable
            variable = variable.next_step_variables

    def __len__(self):
        num_steps = 1
        variable = self.next_step_variables
        while variable is not None:
            num_steps += 1
            variable = variable.next_step_variables
        return num_steps

    def is_initial_step(self) -> bool:
        return self.prev_step_variables is None

    def step_index(self):
        if self._prev_step_variables is None:
            return 0
        else:
            return 1 + self._prev_step_variables.step_index()

    def get_variables(self, depth: int = 0) -> Dict[str, nn.Variable]:
        variables = {}

        prefix = f"step_{depth}_" if depth > 0 else ""

        def _append_variable(name: str, variable: nn.Variable) -> None:
            if variable is None:
                return
            if isinstance(variable, nn.Variable):
                variables[f"{prefix}{name}"] = variable
            elif isinstance(variable, (list, tuple)):
                for i, v in enumerate(variable):
                    _append_variable(f"{prefix}{name}_{i}", v)
            else:
                raise ValueError(f"invalid variable type: {type(variable)}")

        # add standard variables
        if hasattr(self, "s_current"):
            _append_variable("s_current", self.s_current)
        if hasattr(self, "a_current"):
            _append_variable("a_current", self.a_current)
        if hasattr(self, "reward"):
            _append_variable("reward", self.reward)
        if hasattr(self, "gamma"):
            _append_variable("gamma", self.gamma)
        if hasattr(self, "non_terminal"):
            _append_variable("non_terminal", self.non_terminal)
        if hasattr(self, "s_next"):
            _append_variable("s_next", self.s_next)
        if hasattr(self, "weight"):
            _append_variable("weight", self.weight)

        # recursively append next step variables
        if self.next_step_variables:
            next_step_variables = self.next_step_variables.get_variables(depth + 1)
            variables.update(next_step_variables)

        # add extra variables
        for name, variable in self.extra.items():
            _append_variable(name, variable)

        # add rnn state variables
        for state_name, state in self.rnn_states.items():
            for variable_name, variable in state.items():
                _append_variable(f"{state_name}_{variable_name}", variable)

        return variables


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
        self._assert_no_duplicate_model(self._models)
        if self._need_rnn_support(self._models) and not self.support_rnn():
            raise NotImplementedError(f'{self.__name__} does not support RNN models!')
        self._solvers = solvers

        # Initially create training variables with batch_size 1.
        # The batch_size will be updated later depending on the given experience data
        # This procedure is a workaround to initialize model parameters (it it is not created).
        total_timesteps = self._config.unroll_steps + self._config.burn_in_steps
        next_step_variables = None
        for _ in range(total_timesteps):
            training_variables = self._setup_training_variables(1)
            training_variables.next_step_variables = next_step_variables
            next_step_variables = training_variables
        self._training_variables = training_variables
        self._assert_variable_length_equals_total_timesteps()

        self._build_training_graph(self._models, self._training_variables)

        self._setup_solver()

    @property
    def __name__(self):
        return self.__class__.__name__

    def train(self, batch: TrainingBatch, **kwargs) -> Dict[str, np.ndarray]:
        if self._models is None:
            raise RuntimeError('Call setup_training() first. Model is not set!')
        self._train_count += 1

        batch = self._setup_batch(batch)
        new_batch_size = batch.batch_size
        prev_batch_size = self._training_variables.batch_size
        if new_batch_size != prev_batch_size:
            total_timesteps = self._config.unroll_steps + self._config.burn_in_steps
            assert 0 < total_timesteps
            next_step_variables = None
            for _ in range(total_timesteps):
                training_variables = self._setup_training_variables(new_batch_size)
                training_variables.next_step_variables = next_step_variables
                next_step_variables = training_variables
            self._training_variables = training_variables
            self._assert_variable_length_equals_total_timesteps()

            self._build_training_graph(self._models, self._training_variables)

        trainer_state = self._update_model(self._models, self._solvers, batch, self._training_variables, **kwargs)

        return trainer_state

    @property
    @abstractmethod
    def loss_variables(self) -> Dict[str, nn.Variable]:
        raise NotImplementedError

    @property
    def training_variables(self) -> TrainingVariables:
        return self._training_variables

    def set_learning_rate(self, new_learning_rate):
        for solver in self._solvers.values():
            solver.set_learning_rate(new_learning_rate)

    def support_rnn(self) -> bool:
        return False

    def _setup_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        return training_batch

    @abstractmethod
    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
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

    def _assert_variable_length_equals_total_timesteps(self):
        total_timesptes = self._config.unroll_steps + self._config.burn_in_steps
        if len(self._training_variables) != total_timesptes:
            raise RuntimeError(f'Training variables length and rnn unroll + burn-in steps does not match!. \
                                   {len(self._training_variables)} != {total_timesptes}. \
                                   Check that the training method supports recurrent networks.')

    @classmethod
    def _assert_no_duplicate_model(cls, models):
        scope_names = set()
        for model in models:
            scope_name = model.scope_name
            assert scope_name not in scope_names
            scope_names.add(scope_name)

    def _need_rnn_support(self, models: Sequence[Model]):
        for model in models:
            if model.is_recurrent():
                return True
        return False
