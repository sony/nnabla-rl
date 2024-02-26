# Copyright 2024 Sony Group Corporation.
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
from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.policy.soft_policy_trainer import AdjustableTemperature
from nnabla_rl.models import Model, QFunction, StochasticPolicy
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable, sync_model


@dataclass
class SRSACActorCriticTrainerConfig(TrainerConfig):
    fixed_temperature: bool = False
    target_entropy: Optional[float] = None
    replay_ratio: int = 1
    tau: float = 0.005

    def __post_init__(self):
        super().__post_init__()


class SRSACActorCriticTrainer(ModelTrainer):
    """Efficient implementation of SAC style training that trains a policy and
    a q-function in parallel."""
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _q_solver: nn.solver.Solver
    _q_functions: Sequence[QFunction]
    _target_q_functions: Sequence[QFunction]
    _q_losses: Sequence[nn.Variable]
    _pi_solver: nn.solver.Solver
    _pi_losses: Sequence[nn.Variable]
    _temperature: AdjustableTemperature
    _temperature_solver: Optional[nn.solver.Solver]
    _temperature_losses: Sequence[nn.Variable]
    _config: SRSACActorCriticTrainerConfig

    def __init__(self,
                 pi: StochasticPolicy,
                 pi_solvers: Dict[str, nn.solver.Solver],
                 q_functions: Tuple[QFunction, QFunction],
                 q_solvers: Dict[str, nn.solver.Solver],
                 target_q_functions: Tuple[QFunction, QFunction],
                 temperature: AdjustableTemperature,
                 temperature_solver: Optional[nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SRSACActorCriticTrainerConfig = SRSACActorCriticTrainerConfig()):
        if len(q_functions) != 2:
            raise ValueError('Two q functions should be provided')
        if not config.fixed_temperature and temperature_solver is None:
            raise ValueError('Please set solver for temperature model')
        self._pi_solver = pi_solvers[pi.scope_name]
        self._q_functions = q_functions
        self._q_solvers = [q_solvers[q_function.scope_name] for q_function in q_functions]
        self._target_q_functions = target_q_functions
        self._temperature = temperature
        self._temperature_solver = temperature_solver
        if config.target_entropy is None:
            config.target_entropy = -env_info.action_dim
        models = (pi, *q_functions)
        solvers = {}
        solvers.update(pi_solvers)
        solvers.update(q_solvers)
        super().__init__(models, solvers, env_info, config)

    def support_rnn(self) -> bool:
        return False

    def _total_timesteps(self) -> int:
        return self._config.replay_ratio

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

        # update model
        for pi_loss, q_loss, temperature_loss in zip(self._pi_losses, self._q_losses, self._temperature_losses):
            self._q_training(q_loss, self._q_solvers)
            for q, target_q in zip(self._q_functions, self._target_q_functions):
                sync_model(q, target_q, tau=self._config.tau)
            self._pi_training(pi_loss, self._pi_solver, temperature_loss, self._temperature_solver)

        trainer_state = {}
        trainer_state['pi_loss'] = self._pi_losses[-1].d.copy()
        trainer_state['td_errors'] = self._td_errors[-1].d.copy()
        trainer_state['q_loss'] = self._q_losses[-1].d.copy()
        trainer_state['temperature_loss'] = self._temperature_losses[-1].d.copy()
        return trainer_state

    def _q_training(self, q_loss, q_solvers):
        for solver in q_solvers:
            solver.zero_grad()
        q_loss.forward()
        q_loss.backward()
        for solver in q_solvers:
            solver.update()

    def _pi_training(self, pi_loss, pi_solver, temperature_loss, temperature_solver):
        pi_solver.zero_grad()
        pi_loss.forward()
        pi_loss.backward()
        pi_solver.update()
        # Update temperature if requested
        if not self._config.fixed_temperature:
            assert temperature_solver is not None
            assert temperature_loss is not None
            temperature_solver.zero_grad()
            temperature_loss.forward()
            temperature_loss.backward()
            temperature_solver.update()

    def get_temperature(self) -> nn.Variable:
        # Will return exponentiated log temperature. To keep temperature always positive
        return self._temperature()

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        self._q_losses = []
        self._td_errors = []
        self._pi_losses = []
        self._temperature_losses = []
        for _, variables in enumerate(training_variables):
            q_loss, td_error, pi_loss, temperature_loss = self._build_one_step_graph(models, variables)
            self._q_losses.append(q_loss)
            self._td_errors.append(td_error)
            self._pi_losses.append(pi_loss)
            self._temperature_losses.append(temperature_loss)

    def _build_one_step_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        policy = cast(StochasticPolicy, models[0])
        q_functions = cast(Tuple[QFunction, QFunction], models[1:])
        target_q_functions = self._target_q_functions
        assert len(q_functions) == len(target_q_functions)
        for q_function in q_functions:
            assert isinstance(q_function, QFunction)
        q_loss, td_error = self._build_q_training_graph(q_functions, target_q_functions, policy, training_variables)

        assert isinstance(policy, StochasticPolicy)
        pi_loss, temperature_loss = self._build_policy_training_graph(policy, q_functions, training_variables)
        return q_loss, td_error, pi_loss, temperature_loss

    def _build_q_training_graph(self,
                                q_functions: Sequence[QFunction],
                                target_q_functions: Sequence[QFunction],
                                target_policy: StochasticPolicy,
                                training_variables: TrainingVariables) -> Tuple[nn.Variable, nn.Variable]:
        # NOTE: Target q value depends on underlying implementation
        target_q = self._compute_q_target(target_q_functions, target_policy, training_variables)
        target_q.need_grad = False

        q_loss = 0
        for model in q_functions:
            loss, extra = self._compute_squared_td_loss(model, target_q, training_variables)
            q_loss += loss

        # FIXME: using the last q function's td error for prioritized replay. Is this fine?
        td_error = extra['td_error']
        td_error.persistent = True

        return q_loss, td_error

    def _compute_squared_td_loss(self,
                                 model: QFunction,
                                 target_q: nn.Variable,
                                 training_variables: TrainingVariables) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        s_current = training_variables.s_current
        a_current = training_variables.a_current

        td_error = target_q - model.q(s_current, a_current)

        squared_td_error = training_variables.weight * NF.pow_scalar(td_error, 2.0)
        q_loss = NF.mean(squared_td_error)

        extra = {'td_error': td_error}
        return q_loss, extra

    def _compute_q_target(self,
                          target_q_functions: Sequence[QFunction],
                          target_policy: StochasticPolicy,
                          training_variables: TrainingVariables,
                          **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        policy_distribution = target_policy.pi(s_next)
        a_next, log_pi = policy_distribution.sample_and_compute_log_prob()

        q_values = []
        for target_q_function in target_q_functions:
            q_value = target_q_function.q(s_next, a_next)
            q_values.append(q_value)

        target_q = RF.minimum_n(q_values)
        return reward + gamma * non_terminal * (target_q - self.get_temperature() * log_pi)

    def _build_policy_training_graph(self, policy: StochasticPolicy,
                                     q_functions: Sequence[QFunction],
                                     training_variables: TrainingVariables) -> Tuple[nn.Variable, nn.Variable]:
        # Actor optimization graph
        policy_distribution = policy.pi(training_variables.s_current)
        action_var, log_pi = policy_distribution.sample_and_compute_log_prob()
        q_values = []
        for q_function in q_functions:
            q_values.append(q_function.q(training_variables.s_current, action_var))
        min_q = RF.minimum_n(q_values)
        pi_loss = NF.mean(self.get_temperature() * log_pi - min_q)

        if not self._config.fixed_temperature:
            assert isinstance(log_pi, nn.Variable)
            log_pi_unlinked = log_pi.get_unlinked_variable()
            temperature_loss = -NF.mean(self.get_temperature() * (log_pi_unlinked + self._config.target_entropy))
        return pi_loss, temperature_loss

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        s_next_var = create_variable(batch_size, self._env_info.state_shape)
        reward_var = create_variable(batch_size, 1)
        gamma_var = create_variable(batch_size, 1)
        non_terminal_var = create_variable(batch_size, 1)
        weight_var = create_variable(batch_size, 1)

        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               a_current=a_current_var,
                                               reward=reward_var,
                                               gamma=gamma_var,
                                               non_terminal=non_terminal_var,
                                               s_next=s_next_var,
                                               weight=weight_var)
        return training_variables

    def _setup_solver(self):
        super()._setup_solver()
        if not self._config.fixed_temperature:
            self._temperature_solver.set_parameters(self._temperature.get_parameters(), reset=False, retain_state=True)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_losses[-1],
                "q_loss": self._q_losses[-1],
                "temperature_loss": self._temperature_losses[-1]}
