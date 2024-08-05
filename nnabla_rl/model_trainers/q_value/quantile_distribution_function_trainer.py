# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.logger import logger
from nnabla_rl.model_trainers.model_trainer import LossIntegration, TrainingBatch, TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.multi_step_trainer import MultiStepTrainer, MultiStepTrainerConfig
from nnabla_rl.models import Model, QuantileDistributionFunction
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables


@dataclass
class QuantileDistributionFunctionTrainerConfig(MultiStepTrainerConfig):
    num_quantiles: int = 200
    kappa: float = 1.0


class QuantileDistributionFunctionTrainer(MultiStepTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: QuantileDistributionFunctionTrainerConfig
    _tau_hat_var: nn.Variable
    _quantile_huber_loss: nn.Variable
    _prev_rnn_states: Dict[str, Dict[str, nn.Variable]]

    def __init__(
        self,
        models: Union[QuantileDistributionFunction, Sequence[QuantileDistributionFunction]],
        solvers: Dict[str, nn.solver.Solver],
        env_info: EnvironmentInfo,
        config: QuantileDistributionFunctionTrainerConfig = QuantileDistributionFunctionTrainerConfig(),
    ):
        if config.kappa == 0.0:
            logger.info("kappa is set to 0.0. Quantile regression loss will be used for training")
        else:
            logger.info("kappa is non 0.0. Quantile huber loss will be used for training")

        tau_hat = self._precompute_tau_hat(config.num_quantiles)
        self._tau_hat_var = nn.Variable.from_numpy_array(tau_hat)
        self._prev_rnn_states = {}
        super(QuantileDistributionFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(
        self,
        models: Sequence[Model],
        solvers: Dict[str, nn.solver.Solver],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.reward, b.reward)
            set_data_to_variable(t.gamma, b.gamma)
            set_data_to_variable(t.non_terminal, b.non_terminal)
            set_data_to_variable(t.s_next, b.s_next)

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
        self._quantile_huber_loss.forward()
        self._quantile_huber_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state["q_loss"] = self._quantile_huber_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        self._quantile_huber_loss = 0
        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        models = cast(Sequence[QuantileDistributionFunction], models)

        # Ttheta_j is the target quantile distribution
        Ttheta_j = self._compute_target(training_variables)
        Ttheta_j = RF.expand_dims(Ttheta_j, axis=1)
        Ttheta_j.need_grad = False

        prev_rnn_states = self._prev_rnn_states
        train_rnn_states = training_variables.rnn_states
        for model in models:
            with rnn_support(model, prev_rnn_states, train_rnn_states, training_variables, self._config):
                loss, _ = self._compute_loss(model, Ttheta_j, training_variables)
            self._quantile_huber_loss += 0.0 if ignore_loss else loss

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        raise NotImplementedError

    def _compute_loss(
        self, model: QuantileDistributionFunction, target: nn.Variable, training_variables: TrainingVariables
    ) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        batch_size = training_variables.batch_size
        Ttheta_i = model.quantiles(training_variables.s_current, training_variables.a_current)
        Ttheta_i = RF.expand_dims(Ttheta_i, axis=2)
        assert Ttheta_i.shape == (batch_size, self._config.num_quantiles, 1)

        tau_hat = RF.expand_dims(self._tau_hat_var, axis=0)
        tau_hat = RF.repeat(tau_hat, repeats=batch_size, axis=0)
        tau_hat = RF.expand_dims(tau_hat, axis=2)
        assert tau_hat.shape == Ttheta_i.shape

        # NOTE: target is same as Ttheta_j in the paper
        quantile_huber_loss = RF.quantile_huber_loss(target, Ttheta_i, self._config.kappa, tau_hat)
        assert quantile_huber_loss.shape == (batch_size, self._config.num_quantiles, self._config.num_quantiles)

        quantile_huber_loss = NF.mean(quantile_huber_loss, axis=2)
        quantile_huber_loss = NF.sum(quantile_huber_loss, axis=1)
        return NF.mean(quantile_huber_loss), {}

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        s_next_var = create_variable(batch_size, self._env_info.state_shape)
        reward_var = create_variable(batch_size, 1)
        gamma_var = create_variable(batch_size, 1)
        non_terminal_var = create_variable(batch_size, 1)

        rnn_states = {}
        for model in self._models:
            if model.is_recurrent():
                rnn_state_variables = create_variables(batch_size, model.internal_state_shapes())
                rnn_states[model.scope_name] = rnn_state_variables

        training_variables = TrainingVariables(
            batch_size=batch_size,
            s_current=s_current_var,
            a_current=a_current_var,
            reward=reward_var,
            gamma=gamma_var,
            non_terminal=non_terminal_var,
            s_next=s_next_var,
            rnn_states=rnn_states,
        )

        return training_variables

    @staticmethod
    def _precompute_tau_hat(num_quantiles):
        tau_hat = [
            (tau_prev + tau_i) / num_quantiles / 2.0
            for tau_prev, tau_i in zip(range(0, num_quantiles), range(1, num_quantiles + 1))
        ]
        return np.array(tau_hat, dtype=np.float32)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"quantile_huber_loss": self._quantile_huber_loss}
