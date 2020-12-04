from typing import Iterable, Dict

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import VFunction, Model


@dataclass
class SquaredTDVFunctionTrainerParam(TrainerParam):
    reduction_method: str = 'mean'
    v_loss_scalar: float = 1.0

    def __post_init__(self):
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')


class SquaredTDVFunctionTrainer(ModelTrainer):
    def __init__(self, env_info: EnvironmentInfo,
                 params: SquaredTDVFunctionTrainerParam):
        super(SquaredTDVFunctionTrainer, self).__init__(env_info, params)

        # Training loss/output
        self._v_loss = None

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      experience,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict:
        (s, *_) = experience

        training_variables.s_current.d = s

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._v_loss.forward(clear_no_need_grad=True)
        self._v_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        errors = {}
        return errors

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        for model in models:
            assert isinstance(model, VFunction)
        # value function learning
        target_v = training.compute_target(training_variables)

        td_errors = [target_v - v_function.v(training_variables.s_current) for v_function in models]
        v_loss = 0
        for td_error in td_errors:
            squared_td_error = NF.pow_scalar(td_error, 2.0)
            if self._params.reduction_method == 'mean':
                v_loss += self._params.v_loss_scalar * NF.mean(squared_td_error)
            elif self._params.reduction_method == 'sum':
                v_loss += self._params.v_loss_scalar * NF.sum(squared_td_error)
            else:
                raise RuntimeError
        self._v_loss = v_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        training_variables = TrainingVariables(s_current_var)
        return training_variables
