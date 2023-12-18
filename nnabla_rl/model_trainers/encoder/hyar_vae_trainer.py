# Copyright 2023 Sony Group Corporation.
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
from typing import Dict, Iterable, Sequence, Union, cast

import gym
import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RNF
from nnabla_rl.distributions import Gaussian
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables)
from nnabla_rl.models import HyARVAE, Model
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class HyARVAETrainerConfig(TrainerConfig):
    beta: float = 1.0


class HyARVAETrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: HyARVAETrainerConfig
    _encoder_loss: nn.Variable  # Training loss/output

    def __init__(self,
                 models: Union[HyARVAE, Sequence[HyARVAE]],
                 solvers: Dict[str, nn.solver.Solver],
                 env_info: EnvironmentInfo,
                 config: HyARVAETrainerConfig = HyARVAETrainerConfig()):
        super().__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Iterable[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict[str, np.ndarray]:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)
            set_data_to_variable(t.a_current, b.a_current)
            set_data_to_variable(t.s_next, b.s_next)

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._encoder_loss.forward(clear_no_need_grad=True)
        self._encoder_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['encoder_loss'] = self._encoder_loss.d.copy()
        trainer_state['kl_loss'] = self._kl_loss.d.copy()
        trainer_state['reconstruction_loss'] = self._reconstruction_loss.d.copy()
        trainer_state['dyn_loss'] = self._dyn_loss.d.copy()
        return trainer_state

    def _build_training_graph(self, models: Union[Model, Sequence[Model]], training_variables: TrainingVariables):
        self._encoder_loss = 0
        models = cast(Sequence[HyARVAE], models)
        ignore_intermediate_loss = self._config.loss_integration is LossIntegration.LAST_TIMESTEP_ONLY
        for step_index, variables in enumerate(training_variables):
            is_burn_in_steps = step_index < self._config.burn_in_steps
            is_intermediate_steps = step_index < self._config.burn_in_steps + self._config.unroll_steps - 1
            ignore_loss = is_burn_in_steps or (is_intermediate_steps and ignore_intermediate_loss)
            self._build_one_step_graph(models, variables, ignore_loss=ignore_loss)

    def _build_one_step_graph(self,
                              models: Sequence[Model],
                              training_variables: TrainingVariables,
                              ignore_loss: bool):
        batch_size = training_variables.batch_size

        models = cast(Sequence[HyARVAE], models)
        for vae in models:
            action1, action2 = training_variables.a_current
            action_space = cast(gym.spaces.Tuple, self._env_info.action_space)
            xk = action1 if isinstance(action_space[0], gym.spaces.Box) else action2
            latent_distribution, (xk_tilde, ds_tilde) = vae.encode_and_decode(x=xk,
                                                                              state=training_variables.s_current,
                                                                              action=training_variables.a_current)

            latent_shape = (batch_size, latent_distribution.ndim)
            target_latent_distribution = Gaussian(
                mean=nn.Variable.from_numpy_array(np.zeros(shape=latent_shape, dtype=np.float32)),
                ln_var=nn.Variable.from_numpy_array(np.zeros(shape=latent_shape, dtype=np.float32))
            )

            reconstruction_loss = RNF.mean_squared_error(xk, xk_tilde)
            kl_divergence = latent_distribution.kl_divergence(target_latent_distribution)
            latent_loss = NF.mean(kl_divergence)

            ds = cast(nn.Variable, training_variables.s_next) - cast(nn.Variable, training_variables.s_current)
            prediction_loss = RNF.mean_squared_error(ds, ds_tilde)
            assert reconstruction_loss.shape == latent_loss.shape
            assert reconstruction_loss.shape == prediction_loss.shape
            # dividing by ndim to match author's code
            kl_loss = 0.5 * latent_loss / latent_distribution.ndim
            reconstruction_loss = 2.0 * reconstruction_loss
            dyn_loss = self._config.beta * prediction_loss

            self._kl_loss = kl_loss
            self._reconstruction_loss = reconstruction_loss
            self._dyn_loss = dyn_loss
            self._kl_loss.persistent = True
            self._reconstruction_loss.persistent = True
            self._dyn_loss.persistent = True

            self._encoder_loss += 0.0 if ignore_loss else kl_loss + reconstruction_loss + dyn_loss

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        a_current_var = create_variable(batch_size, self._env_info.action_shape)
        s_next_var = create_variable(batch_size, self._env_info.state_shape)

        return TrainingVariables(batch_size, s_current_var, a_current_var, s_next=s_next_var)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"encoder_loss": self._encoder_loss}

    def support_rnn(self) -> bool:
        # TODO: support rnn model
        return False
