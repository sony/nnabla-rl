from typing import Iterable, Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

import nnabla_rl.functions as RF
from nnabla_rl.logger import logger
from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import QuantileDistributionFunction, Model


@dataclass
class QRDQNQuantileDistributionFunctionTrainerParam(TrainerParam):
    gamma: float = 0.99
    num_quantiles: int = 200
    kappa: float = 1.0


class QRDQNQuantileDistributionFunctionTrainer(ModelTrainer):
    def __init__(self,
                 env_info,
                 params: QRDQNQuantileDistributionFunctionTrainerParam):
        super(QRDQNQuantileDistributionFunctionTrainer, self).__init__(env_info, params)
        if self._params.kappa == 0.0:
            logger.info("kappa is set to 0.0. Quantile regression loss will be used for training")
        else:
            logger.info("kappa is non 0.0. Quantile huber loss will be used for training")

        tau_hat = self._precompute_tau_hat(self._params.num_quantiles)
        self._tau_hat_var = nn.Variable.from_numpy_array(tau_hat)

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
            assert isinstance(model, QuantileDistributionFunction)
        batch_size = training_variables.batch_size

        # Ttheta_j is the target quantile distribution
        Ttheta_j = self._training.compute_target(training_variables)
        Ttheta_j = RF.expand_dims(Ttheta_j, axis=1)
        Ttheta_j.need_grad = False

        self._quantile_huber_loss = 0
        for model in models:
            Ttheta_i = model.quantiles(s=training_variables.s_current)
            Ttheta_i = model._quantiles_of(Ttheta_i, training_variables.a_current)
            Ttheta_i = RF.expand_dims(Ttheta_i, axis=2)
            assert Ttheta_i.shape == (batch_size, self._params.num_quantiles, 1)

            tau_hat = RF.expand_dims(self._tau_hat_var, axis=0)
            tau_hat = RF.repeat(tau_hat, repeats=batch_size, axis=0)
            tau_hat = RF.expand_dims(tau_hat, axis=2)
            assert tau_hat.shape == Ttheta_i.shape

            quantile_huber_loss = RF.quantile_huber_loss(Ttheta_j, Ttheta_i, self._params.kappa, tau_hat)
            assert quantile_huber_loss.shape == (batch_size,
                                                 self._params.num_quantiles,
                                                 self._params.num_quantiles)

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

    def _precompute_tau_hat(self, num_quantiles):
        tau_hat = [(tau_prev + tau_i) / num_quantiles / 2.0
                   for tau_prev, tau_i in zip(range(0, num_quantiles), range(1, num_quantiles+1))]
        return np.array(tau_hat, dtype=np.float32)
