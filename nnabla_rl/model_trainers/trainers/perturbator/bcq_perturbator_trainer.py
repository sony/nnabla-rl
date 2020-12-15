from typing import Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import VariationalAutoEncoder, Perturbator, QFunction, Model


@dataclass
class BCQPerturbatorTrainerParam(TrainerParam):
    '''
    Args:
        phi(float): action perturbator noise coefficient
    '''
    phi: float = 0.05


class BCQPerturbatorTrainer(ModelTrainer):
    def __init__(self, env_info: EnvironmentInfo,
                 params: BCQPerturbatorTrainerParam,
                 q_function: QFunction,
                 vae: VariationalAutoEncoder):
        super(BCQPerturbatorTrainer, self).__init__(env_info, params)
        self._q_function = q_function
        self._vae = vae

        # Training loss/output
        self._perturbator_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict:
        training_variables.s_current.d = batch.s_current

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._perturbator_loss.forward(clear_no_need_grad=True)
        self._perturbator_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        errors = {}
        return errors

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        for model in models:
            assert isinstance(model, Perturbator)
        batch_size = training_variables.batch_size

        self._perturbator_loss = 0
        for perturbator in models:
            action = self._vae.decode(training_variables.s_current)
            action.need_grad = False

            noise = perturbator.generate_noise(training_variables.s_current, action, phi=self._params.phi)

            xi_loss = -self._q_function.q(training_variables.s_current, action + noise)
            assert xi_loss.shape == (batch_size, 1)

            self._perturbator_loss += NF.mean(xi_loss)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        training_variables = TrainingVariables(batch_size, s_current_var)
        return training_variables
