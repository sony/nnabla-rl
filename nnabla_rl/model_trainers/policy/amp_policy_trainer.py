# Copyright 2024 Sony Group Corporation.
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

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.functions import compute_std, normalize
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables)
from nnabla_rl.models import Model, StochasticPolicy
from nnabla_rl.utils.data import add_batch_dimension, set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class AMPPolicyTrainerConfig(TrainerConfig):
    action_bound_loss_coefficient: float = 10.0
    normalize_action: bool = True
    action_mean: Optional[Tuple[float, ...]] = None
    action_var: Optional[Tuple[float, ...]] = None
    epsilon: float = 0.2
    regularization_coefficient: float = 0.0005


class AMPPolicyTrainer(ModelTrainer):
    """Adversarial Motion Prior Policy Trainer."""

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: AMPPolicyTrainerConfig
    _pi_loss: nn.Variable

    def __init__(self,
                 models: Union[StochasticPolicy, Sequence[StochasticPolicy]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: AMPPolicyTrainerConfig = AMPPolicyTrainerConfig()):

        self._action_mean = None
        self._action_std = None
        if config.normalize_action:
            action_mean = add_batch_dimension(np.array(config.action_mean, dtype=np.float32))
            self._action_mean = nn.Variable.from_numpy_array(action_mean)
            action_var = add_batch_dimension(np.array(config.action_var, dtype=np.float32))
            self._action_std = compute_std(nn.Variable.from_numpy_array(action_var),
                                           epsilon=0.0, mode_for_floating_point_error="max")

        super(AMPPolicyTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.extra["log_prob"], b.extra["log_prob"])
            set_data_to_variable(t.extra["advantage"], b.extra["advantage"])

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._pi_loss.forward()
        self._pi_loss.backward()
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state["pi_loss"] = self._pi_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Sequence[Model], training_variables: TrainingVariables):
        models = cast(Sequence[StochasticPolicy], models)

        self._pi_loss = 0.0

        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self, models: Sequence[Model], training_variables: TrainingVariables, ignore_loss: bool):
        action_min_bound, action_max_bound = self._compute_action_bound()

        models = cast(Sequence[StochasticPolicy], models)
        for policy in models:
            distribution = policy.pi(training_variables.s_current)

            clip_loss = self._clip_loss(training_variables, distribution)

            action_bound_loss = self._bound_loss(distribution.mean(), action_min_bound, action_max_bound)

            regularization_loss = self._regularization_loss(policy)

            self._pi_loss += 0.0 if ignore_loss else clip_loss + action_bound_loss + regularization_loss

    def _compute_action_bound(self) -> Tuple[nn.Variable, nn.Variable]:
        assert isinstance(self._env_info.action_space, gym.spaces.Box)
        action_min = nn.Variable.from_numpy_array(add_batch_dimension(self._env_info.action_space.low))
        action_max = nn.Variable.from_numpy_array(add_batch_dimension(self._env_info.action_space.high))

        if self._config.normalize_action:
            normalized_action_min = normalize(action_min, mean=self._action_mean, std=self._action_std)
            normalized_action_max = normalize(action_max, mean=self._action_mean, std=self._action_std)
            return normalized_action_min, normalized_action_max
        else:
            return action_min, action_max

    def _clip_loss(self, training_variables: TrainingVariables, distribution: Distribution) -> nn.Variable:
        a_current = training_variables.a_current
        if self._config.normalize_action:
            a_current = normalize(a_current, mean=self._action_mean, std=self._action_std)

        log_prob_new = distribution.log_prob(a_current)
        log_prob_old = training_variables.extra["log_prob"]
        probability_ratio = NF.exp(log_prob_new - log_prob_old)
        clipped_ratio = NF.clip_by_value(probability_ratio, 1 - self._config.epsilon, 1 + self._config.epsilon)
        advantage = training_variables.extra["advantage"]
        lower_bounds = NF.minimum2(probability_ratio * advantage, clipped_ratio * advantage)
        clip_loss = - NF.mean(lower_bounds)
        return clip_loss

    def _bound_loss(self, mean: nn.Variable, bound_min: nn.Variable, bound_max: nn.Variable, axis: int = -1
                    ) -> nn.Variable:
        violation_min = NF.minimum_scalar(mean - bound_min, 0.0)
        violation_max = NF.maximum_scalar(mean - bound_max, 0.0)
        violation = NF.sum((violation_min**2), axis=axis) + NF.sum((violation_max**2), axis=axis)
        loss = 0.5 * NF.mean(violation)
        return self._config.action_bound_loss_coefficient * loss

    def _regularization_loss(self, policy: StochasticPolicy) -> nn.Variable:
        regularization_loss = 0.0
        model_params = policy.get_parameters()
        for n, p in model_params.items():
            # without bias
            if not ("/b" in n):
                regularization_loss += self._config.regularization_coefficient * 0.5 * NF.sum(p**2)
        return regularization_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        log_prob_var = create_variable(batch_size, 1)
        advantage_var = create_variable(batch_size, 1)

        extra = {}
        extra["log_prob"] = log_prob_var
        extra["advantage"] = advantage_var
        return TrainingVariables(batch_size, s_current_var, a_current_var, extra=extra)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"pi_loss": self._pi_loss}
