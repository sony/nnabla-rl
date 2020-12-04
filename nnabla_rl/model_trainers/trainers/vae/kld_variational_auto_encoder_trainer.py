from typing import Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

import numpy as np

from dataclasses import dataclass

import nnabla_rl.functions as RNF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import VariationalAutoEncoder, Model
from nnabla_rl.distributions import Gaussian


@dataclass
class KLDVariationalAutoEncoderTrainerParam(TrainerParam):
    pass


class KLDVariationalAutoEncoderTrainer(ModelTrainer):
    def __init__(self, env_info: EnvironmentInfo,
                 params: KLDVariationalAutoEncoderTrainerParam):
        super(KLDVariationalAutoEncoderTrainer, self).__init__(env_info, params)

        # Training loss/output
        self._vae_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      experience,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict:
        (s, a, *_) = experience

        training_variables.s_current.d = s
        training_variables.a_current.d = a

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._vae_loss.forward(clear_no_need_grad=True)
        self._vae_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        errors = {}
        return errors

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        for model in models:
            assert isinstance(model, VariationalAutoEncoder)
        batch_size = training_variables.batch_size

        self._vae_loss = 0
        for vae in models:
            latent_distribution, reconstructed_action = vae(training_variables.s_current, training_variables.a_current)

            latent_shape = (batch_size, *latent_distribution._data_dim)
            target_latent_distribution = Gaussian(mean=np.zeros(shape=latent_shape, dtype=np.float32),
                                                  ln_var=np.zeros(shape=latent_shape, dtype=np.float32))

            reconstruction_loss = RNF.mean_squared_error(training_variables.a_current, reconstructed_action)
            kl_divergence = latent_distribution.kl_divergence(target_latent_distribution)
            latent_loss = 0.5 * NF.mean(kl_divergence)
            self._vae_loss += reconstruction_loss + latent_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        a_current_var = nn.Variable((batch_size, self._env_info.action_dim))

        training_variables = TrainingVariables(s_current_var, a_current_var)
        return training_variables
