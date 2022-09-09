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
from typing import Dict, Sequence, Tuple, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import LossIntegration, TrainingBatch, TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.multi_step_trainer import MultiStepTrainer, MultiStepTrainerConfig
from nnabla_rl.models import Model, ValueDistributionFunction
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables


@dataclass
class ValueDistributionFunctionTrainerConfig(MultiStepTrainerConfig):
    reduction_method: str = 'mean'
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51

    def __post_init__(self):
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')
        return super().__post_init__()


class ValueDistributionFunctionTrainer(MultiStepTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: ValueDistributionFunctionTrainerConfig
    _model: ValueDistributionFunction
    # Training loss/output
    _kl_loss: nn.Variable
    _cross_entropy_loss: nn.Variable
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(self,
                 models: Union[ValueDistributionFunction, Sequence[ValueDistributionFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: ValueDistributionFunctionTrainerConfig):
        self._prev_rnn_states = {}
        super(ValueDistributionFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.reward, b.reward)
            set_data_to_variable(t.gamma, b.gamma)
            set_data_to_variable(t.non_terminal, b.non_terminal)
            set_data_to_variable(t.s_next, b.s_next)
            set_data_to_variable(t.weight, b.weight)

            for model in models:
                if not model.is_recurrent():
                    continue
                # Check batch keys. Because it can be empty.
                # If batch does not provide rnn states, train with zero initial state.
                if model.scope_name not in batch.rnn_states.keys():
                    continue
                b_rnn_states = b.rnn_states[model.scope_name]
                t_rnn_states = t.rnn_states[model.scope_name]

                for state_name in t_rnn_states.keys():
                    set_data_to_variable(t_rnn_states[state_name], b_rnn_states[state_name])

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
        self._cross_entropy_loss = 0
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
        models = cast(Sequence[ValueDistributionFunction], models)

        # Computing the target probabilities
        mi = self._compute_target(training_variables)
        mi.need_grad = False

        prev_rnn_states = self._prev_rnn_states
        train_rnn_states = training_variables.rnn_states
        for model in models:
            with rnn_support(model, prev_rnn_states, train_rnn_states, training_variables, self._config):
                loss, extra_info = self._compute_loss(model, mi, training_variables)
            # Sum over models
            self._cross_entropy_loss += 0.0 if ignore_loss else loss
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
        atom_probabilities = model.probs(training_variables.s_current, training_variables.a_current)
        atom_probabilities = NF.clip_by_value(atom_probabilities, 1e-10, 1.0)
        cross_entropy = target * NF.log(atom_probabilities)
        assert cross_entropy.shape == (batch_size, self._config.num_atoms)

        kl_loss = -NF.sum(cross_entropy, axis=1, keepdims=True)
        if self._config.reduction_method == 'mean':
            loss = NF.mean(kl_loss * training_variables.weight)
        elif self._config.reduction_method == 'sum':
            loss = NF.sum(kl_loss * training_variables.weight)
        else:
            raise RuntimeError
        extra = {'kl_loss': kl_loss}
        return loss, extra

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        s_next_var = create_variable(batch_size, self._env_info.state_shape)
        reward_var = create_variable(batch_size, 1)
        gamma_var = create_variable(batch_size, 1)
        non_terminal_var = create_variable(batch_size, 1)
        weight_var = create_variable(batch_size, 1)

        rnn_states = {}
        for model in self._models:
            if model.is_recurrent():
                rnn_state_variables = create_variables(batch_size, model.internal_state_shapes())
                rnn_states[model.scope_name] = rnn_state_variables

        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               a_current=a_current_var,
                                               reward=reward_var,
                                               gamma=gamma_var,
                                               non_terminal=non_terminal_var,
                                               s_next=s_next_var,
                                               weight=weight_var,
                                               rnn_states=rnn_states)
        return training_variables

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"cross_entropy_loss": self._cross_entropy_loss}
