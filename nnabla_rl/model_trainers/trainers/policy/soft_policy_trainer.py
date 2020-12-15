from typing import Iterable, Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

import nnabla_rl.functions as RNF
from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import Model, QFunction, StochasticPolicy


class AdjustableTemperature(Model):
    def __init__(self, scope_name, initial_value=None):
        super(AdjustableTemperature, self).__init__(scope_name=scope_name)
        if initial_value:
            initial_value = np.log(initial_value)
        else:
            initial_value = np.random.normal()

        initializer = np.reshape(initial_value, newshape=(1, 1))
        with nn.parameter_scope(scope_name):
            self._log_temperature = nn.parameter.get_parameter_or_create(name='log_temperature',
                                                                         shape=(1, 1),
                                                                         initializer=initializer)

    def __call__(self):
        return NF.exp(self._log_temperature)


@dataclass
class SoftPolicyTrainerParam(TrainerParam):
    fixed_temperature: bool = False
    target_entropy: float = None

    def __post_init__(self):
        super(SoftPolicyTrainerParam, self).__post_init__()


class SoftPolicyTrainer(ModelTrainer):
    '''Soft Policy Gradient style Policy Trainer
    '''

    def __init__(self,
                 env_info: EnvironmentInfo,
                 q_functions: Iterable[QFunction],
                 temperature: AdjustableTemperature,
                 temperature_solver: nn.solver.Solver = None,
                 params: SoftPolicyTrainerParam = SoftPolicyTrainerParam()):
        super(SoftPolicyTrainer, self).__init__(env_info, params)
        if len(q_functions) < 2:
            raise ValueError('Must provide at least 2 Q-functions for soft-training')
        self._q_functions = q_functions
        if not self._params.fixed_temperature and temperature_solver is None:
            raise ValueError('Please set solver for temperature model')
        self._temperature = temperature
        self._temperature_solver = temperature_solver

        self._pi_loss = None
        self._temperature_loss = None

        if self._params.target_entropy is None:
            self._params.target_entropy = -self._env_info.action_dim

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs):
        training_variables.s_current.d = batch.s_current

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward()
        self._pi_loss.backward()
        for solver in solvers.values():
            solver.update()
        # Update temperature if requested
        if not self._params.fixed_temperature:
            self._temperature_solver.zero_grad()
            self._temperature_loss.forward()
            self._temperature_loss.backward()
            self._temperature_solver.update()
        errors = {}
        return errors

    def get_temperature(self) -> nn.Variable:
        # Will return exponentiated log temperature. To keep temperature always positive
        return self._temperature()

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        self._pi_loss = 0
        for policy in models:
            assert isinstance(policy, StochasticPolicy)
            # Actor optimization graph
            policy_distribution = policy.pi(self._training_variables.s_current)
            action_var, log_pi = policy_distribution.sample_and_compute_log_prob()
            q_values = []
            for q_function in self._q_functions:
                q_values.append(q_function.q(self._training_variables.s_current, action_var))
            min_q = RNF.minimum_n(q_values)
            self._pi_loss += NF.mean(self.get_temperature() * log_pi - min_q)

        if not self._params.fixed_temperature:
            log_pi_unlinked = log_pi.get_unlinked_variable()
            self._temperature_loss = -NF.mean(self.get_temperature() *
                                              (log_pi_unlinked + self._params.target_entropy))

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        return TrainingVariables(batch_size, s_current_var)

    def _setup_solver(self):
        super()._setup_solver()
        if not self._params.fixed_temperature:
            self._temperature_solver.set_parameters(self._temperature.get_parameters(), reset=False, retain_state=True)
