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
from typing import Dict, Sequence, Tuple, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, ValueDistributionFunction


@dataclass
class ValueDistributionFunctionTrainerConfig(TrainerConfig):
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51


class ValueDistributionFunctionTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ValueDistributionFunctionTrainerConfig
    _model: ValueDistributionFunction
    # Training loss/output
    _kl_loss: nn.Variable
    _cross_entropy_loss: nn.Variable

    def __init__(self,
                 models: Union[ValueDistributionFunction, Sequence[ValueDistributionFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: ValueDistributionFunctionTrainerConfig):
        super(ValueDistributionFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current
        training_variables.reward.d = batch.reward
        training_variables.non_terminal.d = batch.non_terminal
        training_variables.gamma.d = batch.gamma
        training_variables.s_next.d = batch.s_next
        training_variables.weight.d = batch.weight

        for solver in solvers.values():
            solver.zero_grad()
        self._cross_entropy_loss.forward()
        self._cross_entropy_loss.backward()
        for solver in solvers.values():
            solver.update()
        trainer_state = {}
        # Kullbuck Leibler divergence is not actually the td_error itself
        # but is used for prioritizing the replay buffer and we save it as 'td_error' for convenience
        # See: https://arxiv.org/pdf/1710.02298.pdf
        trainer_state['td_errors'] = self._kl_loss.d.copy()
        trainer_state['cross_entropy_loss'] = float(self._cross_entropy_loss.d.copy())
        return trainer_state

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        models = cast(Sequence[ValueDistributionFunction], models)

        # Computing the target probabilities
        mi = self._compute_target(training_variables)
        mi.need_grad = False

        self._cross_entropy_loss = 0
        for model in models:
            loss, extra_info = self._compute_loss(model, mi, training_variables)
            # Sum over models
            self._cross_entropy_loss += loss
        # for prioritized experience replay
        # See: https://arxiv.org/pdf/1710.02298.pdf
        # keep kl_loss only for the last model for prioritized replay
        kl_loss = extra_info['kl_loss']
        self._kl_loss = kl_loss
        self._kl_loss.persistent = True

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        raise NotImplementedError

    def _compute_loss(self,
                      model: ValueDistributionFunction,
                      target: nn.Variable,
                      training_variables: TrainingVariables) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        batch_size = training_variables.batch_size
        atom_probabilities = model.probs(self._training_variables.s_current, self._training_variables.a_current)
        atom_probabilities = NF.clip_by_value(atom_probabilities, 1e-10, 1.0)
        cross_entropy = target * NF.log(atom_probabilities)
        assert cross_entropy.shape == (batch_size, self._config.num_atoms)

        kl_loss = -NF.sum(cross_entropy, axis=1, keepdims=True)
        loss = NF.mean(kl_loss * training_variables.weight)

        extra = {'kl_loss': kl_loss}
        return loss, extra

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        a_current_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))
        reward_var = nn.Variable((batch_size, 1))
        gamma_var = nn.Variable((1, 1))
        non_terminal_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))
        weight_var = nn.Variable((batch_size, 1))
        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               a_current=a_current_var,
                                               reward=reward_var,
                                               gamma=gamma_var,
                                               non_terminal=non_terminal_var,
                                               s_next=s_next_var,
                                               weight=weight_var)
        return training_variables
