from typing import Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.model_trainers.model_trainer import \
    TrainerParam, Training, TrainingBatch, TrainingVariables, ModelTrainer
from nnabla_rl.models import Model, DeterministicPolicy


@dataclass
class DPGPolicyTrainerParam(TrainerParam):
    pass


class DPGPolicyTrainer(ModelTrainer):
    '''Deterministic Policy Gradient (DPG) style Policy Trainer
    '''

    def __init__(self, env_info, q_function, params=DPGPolicyTrainerParam()):
        super(DPGPolicyTrainer, self).__init__(env_info, params)
        self._q_function = q_function
        self._pi_loss = None

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
        self._pi_loss.forward(clear_no_need_grad=True)
        self._pi_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        errors = {}
        return errors

    def _build_training_graph(self, models: Iterable[Model],
                              training: Training,
                              training_variables: TrainingVariables):
        if not isinstance(models[0], DeterministicPolicy):
            raise ValueError

        self._pi_loss = 0
        for policy in models:
            action = policy.pi(training_variables.s_current)
            q = self._q_function.q(training_variables.s_current, action)
            self._pi_loss += -NF.mean(q)

    def _setup_training_variables(self, batch_size):
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        return TrainingVariables(batch_size, s_current_var)
