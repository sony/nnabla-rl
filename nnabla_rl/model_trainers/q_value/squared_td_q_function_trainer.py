# Copyright 2021 Sony Group Corporation.
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
from typing import Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainerConfig, TrainingBatch, TrainingVariables
from nnabla_rl.models import Model, QFunction


@dataclass
class SquaredTDQFunctionTrainerConfig(TrainerConfig):
    reduction_method: str = 'mean'
    grad_clip: Optional[tuple] = None
    q_loss_scalar: float = 1.0

    def __post_init__(self):
        self._assert_one_of(self.reduction_method, ['sum', 'mean'], 'reduction_method')
        if self.grad_clip is not None:
            self._assert_ascending_order(self.grad_clip, 'grad_clip')
            self._assert_length(self.grad_clip, 2, 'grad_clip')


class SquaredTDQFunctionTrainer(ModelTrainer):
    _config: SquaredTDQFunctionTrainerConfig
    # Training loss/output
    _td_error: nn.Variable
    _q_loss: nn.Variable

    def __init__(self,
                 models: Union[QFunction, Sequence[QFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: SquaredTDQFunctionTrainerConfig = SquaredTDQFunctionTrainerConfig()):
        super(SquaredTDQFunctionTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.array]:
        training_variables.s_current.d = batch.s_current
        training_variables.a_current.d = batch.a_current
        training_variables.reward.d = batch.reward
        training_variables.gamma.d = batch.gamma
        training_variables.non_terminal.d = batch.non_terminal
        training_variables.s_next.d = batch.s_next
        training_variables.weight.d = batch.weight

        # update model
        for q_solver in solvers.values():
            q_solver.zero_grad()
        self._q_loss.forward(clear_no_need_grad=True)
        self._q_loss.backward(clear_buffer=True)
        for q_solver in solvers.values():
            q_solver.update()

        trainer_state = {}
        trainer_state['q_loss'] = float(self._q_loss.d.copy())
        trainer_state['td_errors'] = self._td_error.d.copy()
        return trainer_state

    def _build_training_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables):
        models = cast(Sequence[QFunction], models)

        # NOTE: Target q value depends on underlying implementation
        target_q = self._compute_target(training_variables)
        target_q.need_grad = False

        self._q_loss = 0
        for model in models:
            q_loss, extra = self._compute_loss(model, target_q, training_variables)
            self._q_loss += q_loss
        # FIXME: using the last q function's td error for prioritized replay. Is this fine?
        self._td_error = extra['td_error']
        self._td_error.persistent = True

    def _compute_target(self, training_variables: TrainingVariables) -> nn.Variable:
        raise NotImplementedError

    def _compute_loss(self,
                      model: QFunction,
                      target_q: nn.Variable,
                      training_variables: TrainingVariables) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        s_current = training_variables.s_current
        a_current = training_variables.a_current

        td_error = target_q - model.q(s_current, a_current)

        q_loss = 0
        if self._config.grad_clip is not None:
            # NOTE: Gradient clipping is used in DQN and its variants.
            # This operation is same as using huber_loss if the grad_clip value is set to (-1, 1)
            clip_min, clip_max = self._config.grad_clip
            minimum = nn.Variable.from_numpy_array(np.full(td_error.shape, clip_min))
            maximum = nn.Variable.from_numpy_array(np.full(td_error.shape, clip_max))
            td_error = NF.clip_grad_by_value(td_error, minimum, maximum)
        squared_td_error = training_variables.weight * NF.pow_scalar(td_error, 2.0)
        if self._config.reduction_method == 'mean':
            q_loss += self._config.q_loss_scalar * NF.mean(squared_td_error)
        elif self._config.reduction_method == 'sum':
            q_loss += self._config.q_loss_scalar * NF.sum(squared_td_error)
        else:
            raise RuntimeError

        extra = {'td_error': td_error}
        return q_loss, extra

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
        weight_var = nn.Variable((batch_size, 1))

        training_variables = TrainingVariables(batch_size=batch_size,
                                               s_current=s_current_var,
                                               a_current=a_current_var,
                                               reward=reward_var,
                                               gamma=gamma_var,
                                               non_terminal=non_terminal_var,
                                               s_next=s_next_var,
                                               weight=weight_var)
        return training_variables
