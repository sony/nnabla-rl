from typing import Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

import nnabla_rl.functions as RF
from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import StateActionQuantileFunction, Model


@dataclass
class IQNQuantileFunctionTrainerParam(TrainerParam):
    gamma: float = 0.99
    N: int = 64
    N_prime: int = 64
    K: int = 32
    kappa: float = 1.0


class IQNQuantileFunctionTrainer(ModelTrainer):
    def __init__(self,
                 env_info,
                 params: IQNQuantileFunctionTrainerParam):
        super(IQNQuantileFunctionTrainer, self).__init__(env_info, params)

        # Training loss/output
        self._quantile_huber_loss = None

    def train(self, experience, **kwargs) -> Dict:
        (s, a, r, non_terminal, s_next) = experience
        return super().train((s, a, r, self._params.gamma, non_terminal, s_next), **kwargs)

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      experience,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict:
        (s, a, r, gamma, non_terminal, s_next) = experience

        training_variables.s_current.d = s
        training_variables.a_current.d = a
        training_variables.reward.d = r
        training_variables.gamma.d = gamma
        training_variables.non_terminal.d = non_terminal
        training_variables.s_next.d = s_next

        for solver in solvers.values():
            solver.zero_grad()
        self._quantile_huber_loss.forward()
        self._quantile_huber_loss.backward()
        for solver in solvers.values():
            solver.update()

        # TODO: return dictionary of computed errors
        return {}

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        for model in models:
            assert isinstance(model, StateActionQuantileFunction)
        kwargs = {}
        kwargs['K'] = self._params.K
        kwargs['N_prime'] = self._params.N_prime
        batch_size = training_variables.batch_size

        target = training.compute_target(training_variables, **kwargs)
        target = RF.expand_dims(target, axis=1)
        target.need_grad = False
        assert target.shape == (batch_size, 1, self._params.N_prime)

        self._quantile_huber_loss = 0
        for model in models:
            tau_i = model._sample_tau(shape=(batch_size, self._params.N))
            quantiles = model.quantiles(training_variables.s_current, tau_i)
            Z_tau_i = model._quantiles_of(quantiles, training_variables.a_current)
            Z_tau_i = RF.expand_dims(Z_tau_i, axis=2)
            tau_i = RF.expand_dims(tau_i, axis=2)
            assert Z_tau_i.shape == (batch_size, self._params.N, 1)
            assert tau_i.shape == Z_tau_i.shape

            quantile_huber_loss = RF.quantile_huber_loss(target, Z_tau_i, self._params.kappa, tau_i)
            assert quantile_huber_loss.shape == (batch_size, self._params.N, self._params.N_prime)
            quantile_huber_loss = NF.mean(quantile_huber_loss, axis=2)
            quantile_huber_loss = NF.sum(quantile_huber_loss, axis=1)
            self._quantile_huber_loss += NF.mean(quantile_huber_loss)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        a_current_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))
        reward_var = nn.Variable((batch_size, 1))
        gamma_var = nn.Variable((1, 1))
        non_terminal_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))

        training_variables = \
            TrainingVariables(s_current_var, a_current_var, reward_var, gamma_var, non_terminal_var, s_next_var)
        return training_variables
