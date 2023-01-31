# Copyright 2023 Sony Group Corporation.
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
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import DeterministicDecisionTransformer, Model, StochasticDecisionTransformer
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable

DecisionTransformerModel = Union[StochasticDecisionTransformer, DeterministicDecisionTransformer]


@dataclass
class DecisionTransformerTrainerConfig(TrainerConfig):
    context_length: int = 1


@dataclass
class DeterministicDecisionTransformerTrainerConfig(DecisionTransformerTrainerConfig):
    pass


@dataclass
class StochasticDecisionTransformerTrainerConfig(DecisionTransformerTrainerConfig):
    pass


class DecisionTransformerTrainer(ModelTrainer):
    """Decision transformer trainer for Stochastic environment."""
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DecisionTransformerTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[DecisionTransformerModel, Sequence[DecisionTransformerModel]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 wd_solvers: Optional[Dict[str, nn.solver.Solver]],
                 config: DecisionTransformerTrainerConfig):
        self._wd_solvers = {} if wd_solvers is None else wd_solvers
        super(DecisionTransformerTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.extra['timesteps'], b.extra['timesteps'])
            set_data_to_variable(t.extra['rtg'], b.extra['rtg'])
            set_data_to_variable(t.extra['target'], b.extra['target'])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        for wd_solver in self._wd_solvers.values():
            wd_solver.zero_grad()
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()
        for wd_solver in self._wd_solvers.values():
            wd_solver.update()

        trainer_state = {}
        trainer_state['loss'] = self._pi_loss.d.copy()
        return trainer_state

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, (self._config.context_length, *self._env_info.state_shape))
        a_current_var = create_variable(batch_size, (self._config.context_length, *self._env_info.action_shape))
        target_var = create_variable(batch_size, (self._config.context_length, *self._env_info.action_shape))
        timesteps_var = create_variable(batch_size, (1, 1))
        rtg_var = create_variable(batch_size, (self._config.context_length, 1))

        extra = {}
        extra['target'] = target_var
        extra['timesteps'] = timesteps_var
        extra['rtg'] = rtg_var
        return TrainingVariables(batch_size, s_current_var, a_current_var, extra=extra)

    def _setup_solver(self):
        def _should_decay(param_key):
            return 'affine/W' in param_key or 'conv/W' in param_key
        for model in self._models:
            if model.scope_name in self._wd_solvers.keys():
                solver = self._solvers[model.scope_name]
                wd_solver = self._wd_solvers[model.scope_name]

                # wd solver is set
                # Set wd solver to affine/W and Conv2d/W parameters
                all_params = model.get_parameters()
                no_decay_params = {}
                decay_params = {}
                for param_key in all_params.keys():
                    if _should_decay(param_key):
                        decay_params[param_key] = all_params[param_key]
                    else:
                        no_decay_params[param_key] = all_params[param_key]

                # Set retain_state = True and prevent overwriting loaded state (If it is loaded)
                solver.remove_parameters(decay_params.keys())
                solver.set_parameters(no_decay_params, reset=False, retain_state=True)
                wd_solver.remove_parameters(no_decay_params.keys())
                wd_solver.set_parameters(decay_params, reset=False, retain_state=True)
            else:
                solver = self._solvers[model.scope_name]
                # Set retain_state = True and prevent overwriting loaded state (If it is loaded)
                solver.set_parameters(model.get_parameters(), reset=False, retain_state=True)

    def set_learning_rate(self, new_learning_rate):
        for solver in self._solvers.values():
            solver.set_learning_rate(new_learning_rate)
        for solver in self._wd_solvers.values():
            solver.set_learning_rate(new_learning_rate)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}


class StochasticDecisionTransformerTrainer(DecisionTransformerTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: StochasticDecisionTransformerTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[StochasticDecisionTransformer, Sequence[StochasticDecisionTransformer]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 wd_solvers: Optional[Dict[str, nn.solver.Solver]] = None,
                 config: StochasticDecisionTransformerTrainerConfig = StochasticDecisionTransformerTrainerConfig()):
        super(StochasticDecisionTransformerTrainer, self).__init__(models, solvers, env_info, wd_solvers, config)

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticDecisionTransformer], models)
        self._pi_loss = 0
        for policy in models:
            s = training_variables.s_current
            a = training_variables.a_current
            rtg = training_variables.extra['rtg']
            timesteps = training_variables.extra['timesteps']
            target = training_variables.extra['target']
            distribution = policy.pi(s, a, rtg, timesteps)
            # This loss calculation should be same as cross entropy loss
            loss = -distribution.log_prob(target)
            loss = NF.mean(loss)

            self._pi_loss += loss


class DeterministicDecisionTransformerTrainer(DecisionTransformerTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: DeterministicDecisionTransformerTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[DeterministicDecisionTransformer, Sequence[DeterministicDecisionTransformer]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 wd_solvers: Optional[Dict[str, nn.solver.Solver]] = None,
                 config: DeterministicDecisionTransformerTrainerConfig =
                 DeterministicDecisionTransformerTrainerConfig()):
        super(DeterministicDecisionTransformerTrainer, self).__init__(models, solvers, env_info, wd_solvers, config)

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[DeterministicDecisionTransformer], models)
        self._pi_loss = 0
        for policy in models:
            s = training_variables.s_current
            a = training_variables.a_current
            rtg = training_variables.extra['rtg']
            timesteps = training_variables.extra['timesteps']
            target = training_variables.extra['target']
            actions = policy.pi(s, a, rtg, timesteps)
            loss = RF.mean_squared_error(actions, target)
            self._pi_loss += loss
