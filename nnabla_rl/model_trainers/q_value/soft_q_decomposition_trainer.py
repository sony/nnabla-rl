# Copyright 2022 Sony Group Corporation.
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
from typing import Dict, Sequence, Tuple, Union

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingVariables, rnn_support
from nnabla_rl.model_trainers.q_value.soft_q_trainer import SoftQTrainer, SoftQTrainerConfig
from nnabla_rl.models import FactoredContinuousQFunction, QFunction, StochasticPolicy


@dataclass
class SoftQDTrainerConfig(SoftQTrainerConfig):
    pass


class SoftQDTrainer(SoftQTrainer):
    _target_functions: Sequence[FactoredContinuousQFunction]

    def __init__(self,
                 train_functions: Union[FactoredContinuousQFunction, Sequence[FactoredContinuousQFunction]],
                 solvers: Dict[str, nn.solver.Solver],
                 target_functions: Union[FactoredContinuousQFunction, Sequence[FactoredContinuousQFunction]],
                 target_policy: StochasticPolicy,
                 temperature: nn.Variable,
                 env_info: EnvironmentInfo,
                 config: SoftQDTrainerConfig = SoftQDTrainerConfig()):
        super().__init__(
            train_functions=train_functions,
            solvers=solvers,
            target_functions=target_functions,
            target_policy=target_policy,
            temperature=temperature,
            env_info=env_info,
            config=config,
        )

    def _compute_loss(self,
                      model: QFunction,
                      target_q: nn.Variable,
                      training_variables: TrainingVariables) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        assert isinstance(model, FactoredContinuousQFunction)

        s_current = training_variables.s_current
        a_current = training_variables.a_current

        td_error = target_q - model.factored_q(s_current, a_current)

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

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        train_rnn_states = training_variables.rnn_states
        prev_rnn_states = self._prev_target_rnn_states
        with rnn_support(self._target_policy, prev_rnn_states, train_rnn_states, training_variables, self._config):
            policy_distribution = self._target_policy.pi(s_next)
        a_next, log_pi = policy_distribution.sample_and_compute_log_prob()

        factored_q_values = []
        q_values = []
        prev_rnn_states = self._prev_q_rnn_states
        for target_q_function in self._target_functions:
            with rnn_support(target_q_function, prev_rnn_states, train_rnn_states, training_variables, self._config):
                q_value = target_q_function.factored_q(s_next, a_next)
                factored_q_values.append(q_value)
                q_values.append(NF.sum(q_value, axis=1, keepdims=True))

        # get q value minimum index
        stacked_q_values = NF.concatenate(*q_values, axis=1)
        min_indices = RF.argmin(stacked_q_values, axis=1)

        # choose minimum factored q value based on min_indices
        stacked_factored_q_values = NF.stack(*factored_q_values, axis=-1)
        target_q = NF.gather(stacked_factored_q_values, min_indices, axis=2, batch_dims=1)

        # expand reward with entropy bonus
        entropy_bonus = -gamma * self._temperature * log_pi
        reward_with_entropy = NF.concatenate(reward, entropy_bonus, axis=1)

        return reward_with_entropy + gamma * non_terminal * target_q
