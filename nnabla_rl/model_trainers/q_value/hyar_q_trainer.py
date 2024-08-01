# Copyright 2023,2024 Sony Group Corporation.
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
from typing import Any, Dict, Sequence, Tuple, Union, cast

import gym

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch, TrainingVariables
from nnabla_rl.model_trainers.q_value.td3_q_trainer import TD3QTrainer, TD3QTrainerConfig
from nnabla_rl.models import DeterministicPolicy, HyARVAE, Model, QFunction
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class HyARQTrainerConfig(TD3QTrainerConfig):
    noisy_action_max: float = 1.0
    noisy_action_min: float = -1.0
    embed_action_max: float = 1.0
    embed_action_min: float = -1.0
    embed_action_noise_sigma: float = 0.1
    embed_action_noise_abs: float = 1.0
    embed_dim: int = 6
    latent_dim: int = 6


class HyARQTrainer(TD3QTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _target_functions: Sequence[QFunction]
    _target_policy: DeterministicPolicy
    _config: HyARQTrainerConfig

    def __init__(
        self,
        train_functions: Union[QFunction, Sequence[QFunction]],
        solvers: Dict[str, nn.solver.Solver],
        target_functions: Union[QFunction, Sequence[QFunction]],
        target_policy: DeterministicPolicy,
        vae: HyARVAE,
        env_info: EnvironmentInfo,
        config: HyARQTrainerConfig = HyARQTrainerConfig(),
    ):
        self._vae = vae
        super().__init__(train_functions, solvers, target_functions, target_policy, env_info, config)

    def _compute_target(self, training_variables: TrainingVariables, **kwargs) -> nn.Variable:
        gamma = training_variables.gamma
        reward = training_variables.reward
        non_terminal = training_variables.non_terminal
        s_next = training_variables.s_next

        a_next = self._compute_noisy_action(s_next)
        a_next.need_grad = False

        q_values = []
        for target_q_function in self._target_functions:
            q_value = target_q_function.q(s_next, a_next)
            q_values.append(q_value)
        # Use the minimum among computed q_values by default
        target_q = RF.minimum_n(q_values)
        target_q.persistent = True
        return reward + gamma * non_terminal * target_q

    def _compute_noisy_action(self, state):
        a_next_var = self._target_policy.pi(state)
        epsilon = NF.clip_by_value(
            NF.randn(sigma=self._config.train_action_noise_sigma, shape=a_next_var.shape),
            min=-self._config.train_action_noise_abs,
            max=self._config.train_action_noise_abs,
        )
        a_tilde_var = a_next_var + epsilon
        a_tilde_var = NF.clip_by_value(a_tilde_var, self._config.noisy_action_min, self._config.noisy_action_max)
        return a_tilde_var

    def _update_model(
        self,
        models: Sequence[Model],
        solvers: Dict[str, Any],
        batch: TrainingBatch,
        training_variables: TrainingVariables,
        **kwargs,
    ):
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.extra["e"], b.extra["e"])
            set_data_to_variable(t.extra["z"], b.extra["z"])
            set_data_to_variable(t.extra["c_rate"], b.extra["c_rate"])
            set_data_to_variable(t.extra["ds_rate"], b.extra["ds_rate"])
        result = super()._update_model(models, solvers, batch, training_variables, **kwargs)
        return result

    def _compute_loss(
        self, model: QFunction, target_q: nn.Variable, training_variables: TrainingVariables
    ) -> Tuple[nn.Variable, Dict[str, nn.Variable]]:
        e = training_variables.extra["e"]
        z = training_variables.extra["z"]

        e, z = self._reweight_action(e, z, training_variables)
        latent_action = NF.concatenate(e, z)
        latent_action.need_grad = False

        s_current = training_variables.s_current
        q = model.q(s_current, latent_action)
        td_error = target_q - q

        q_loss = 0
        if self._config.loss_type == "squared":
            squared_td_error = training_variables.weight * NF.pow_scalar(td_error, 2.0)
        else:
            raise RuntimeError
        if self._config.reduction_method == "mean":
            q_loss += self._config.q_loss_scalar * NF.mean(squared_td_error)
        else:
            raise RuntimeError
        extra = {"td_error": td_error}
        return q_loss, extra

    def _reweight_action(self, e, z, training_variables: TrainingVariables):
        c_rate = training_variables.extra["c_rate"]
        ds_rate = training_variables.extra["ds_rate"]

        s_current = training_variables.s_current
        s_next = training_variables.s_next
        action1, action2 = training_variables.a_current
        action_space = cast(gym.spaces.Tuple, self._env_info.action_space)
        a_continuous, a_discrete = (
            (action1, action2) if isinstance(action_space[0], gym.spaces.Box) else (action2, action1)
        )

        a_discrete_emb = self._vae.encode_discrete_action(a_discrete)
        a_discrete_emb = NF.clip_by_value(a_discrete_emb, self._config.embed_action_min, self._config.embed_action_max)
        noise = NF.clip_by_value(
            NF.randn(shape=a_discrete_emb.shape) * self._config.embed_action_noise_sigma,
            -self._config.embed_action_noise_abs,
            self._config.embed_action_noise_abs,
        )
        a_discrete_emb_with_noise = NF.clip_by_value(
            a_discrete_emb + noise, self._config.embed_action_min, self._config.embed_action_max
        )

        a_discrete_new = a_discrete
        a_discrete_old = self._vae.decode_discrete_action(e)
        d_mix_rate = NF.equal(a_discrete_new, a_discrete_old)

        e_mixed = d_mix_rate * e + (1.0 - d_mix_rate) * a_discrete_emb_with_noise

        _, ds_model = self._vae.decode(z=z, state=s_current, e=a_discrete_emb)
        ds_data = cast(nn.Variable, s_next) - cast(nn.Variable, s_current)
        actual_ds_rate = NF.reshape(NF.mean(NF.squared_error(ds_model, ds_data), axis=1), shape=(-1, 1))
        actual_ds_rate = NF.abs(actual_ds_rate)
        c_mix_rate = NF.less_equal(actual_ds_rate, ds_rate)

        z_encoded = self._vae.encode(x=a_continuous, state=s_current, e=a_discrete_emb)
        z_encoded = self._apply_c_rate(z_encoded, c_rate)
        z_mixed = c_mix_rate * z + (1.0 - c_mix_rate) * z_encoded

        e_mixed = NF.clip_by_value(e_mixed, self._config.embed_action_min, self._config.embed_action_max)
        z_mixed = NF.clip_by_value(z_mixed, self._config.embed_action_min, self._config.embed_action_max)
        return e_mixed, z_mixed

    def _apply_c_rate(self, z, c_rate):
        median = 0.5 * (c_rate[0] - c_rate[1])
        offset = c_rate[0] - median
        median = NF.reshape(median, shape=(1, -1))
        offset = NF.reshape(offset, shape=(1, -1))
        z = (z - offset) / median
        return z

    def _setup_training_variables(self, batch_size: int) -> TrainingVariables:
        training_variables = super()._setup_training_variables(batch_size)

        extras = {}
        extras["e"] = create_variable(batch_size, (self._config.embed_dim,))
        extras["z"] = create_variable(batch_size, (self._config.latent_dim,))
        extras["ds_rate"] = create_variable(1, (1,))
        extras["c_rate"] = create_variable(1, ((self._config.latent_dim,), (self._config.latent_dim,)))

        training_variables.extra.update(extras)
        return training_variables

    def support_rnn(self) -> bool:
        return False
