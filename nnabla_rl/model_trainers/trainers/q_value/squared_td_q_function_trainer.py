from typing import Iterable, Dict

import numpy as np

import nnabla as nn
import nnabla.functions as NF

from dataclasses import dataclass

from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainerParam, Training, TrainingVariables, ModelTrainer
from nnabla_rl.models import QFunction, Model


@dataclass
class SquaredTDQFunctionTrainerParam(TrainerParam):
    gamma: float = 0.99
    reduction_method: str = 'mean'
    grad_clip: tuple = None
    q_loss_scalar: float = 1.0

    def __post_init__(self):
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')
        if self.grad_clip is not None:
            self._assert_ascending_order(self.grad_clip, 'grad_clip')
            self._assert_length(self.grad_clip, 2, 'grad_clip')


class SquaredTDQFunctionTrainer(ModelTrainer):
    def __init__(self,
                 env_info: EnvironmentInfo,
                 params: SquaredTDQFunctionTrainerParam):
        super(SquaredTDQFunctionTrainer, self).__init__(env_info, params)

        self._weight_var = None

        # Training loss/output
        self._td_error = None
        self._q_loss = None

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
        self._weight_var.d = kwargs['weights']

        # update model
        for q_solver in solvers.values():
            q_solver.zero_grad()
        self._q_loss.forward(clear_no_need_grad=True)
        self._q_loss.backward(clear_buffer=True)
        for q_solver in solvers.values():
            q_solver.update()

        errors = {}
        errors['q_loss'] = self._q_loss.d.copy()
        errors['td_error'] = self._td_error.d.copy()
        return errors

    def _build_training_graph(self,
                              models: Iterable[Model],
                              training: 'Training',
                              training_variables: TrainingVariables):
        for model in models:
            assert isinstance(model, QFunction)

        # NOTE: Target q value depends on selected training
        target_q = training.compute_target(training_variables)
        target_q.need_grad = False

        s_current = training_variables.s_current
        a_current = training_variables.a_current
        td_errors = [target_q - q_function.q(s_current, a_current) for q_function in models]

        q_loss = 0
        for td_error in td_errors:
            if self._params.grad_clip is not None:
                # NOTE: Gradient clipping is used in DQN and its variants.
                # This operation is same as using huber_loss if the grad_clip value is (-1, 1)
                clip_min, clip_max = self._params.grad_clip
                minimum = nn.Variable.from_numpy_array(np.full(td_error.shape, clip_min))
                maximum = nn.Variable.from_numpy_array(np.full(td_error.shape, clip_max))
                td_error = NF.clip_grad_by_value(td_error, minimum, maximum)
            squared_td_error = self._weight_var * NF.pow_scalar(td_error, 2.0)
            if self._params.reduction_method == 'mean':
                q_loss += self._params.q_loss_scalar * NF.mean(squared_td_error)
            elif self._params.reduction_method == 'sum':
                q_loss += self._params.q_loss_scalar * NF.sum(squared_td_error)
            else:
                raise RuntimeError
        self._q_loss = q_loss

        # FIXME: using the last q function's td error for prioritized replay. Is this fine?
        self._td_error = td_error
        self._td_error.persistent = True

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = nn.Variable((batch_size, *self._env_info.state_shape))
        if self._env_info.is_discrete_action_env():
            a_current_var = nn.Variable((batch_size, 1))
        else:
            a_current_var = nn.Variable((batch_size, self._env_info.action_dim))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))
        reward_var = nn.Variable((batch_size, 1))
        gamma_var = nn.Variable((1, 1))
        non_terminal_var = nn.Variable((batch_size, 1))
        s_next_var = nn.Variable((batch_size, *self._env_info.state_shape))

        training_variables = \
            TrainingVariables(s_current_var, a_current_var, reward_var, gamma_var, non_terminal_var, s_next_var)
        self._weight_var = nn.Variable((batch_size, 1))
        return training_variables
