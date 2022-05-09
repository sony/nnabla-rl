# Copyright 2021,2022 Sony Group Corporation.
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
from typing import Dict, Sequence, Union, cast

import nnabla as nn
import nnabla.functions as NF
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import (LossIntegration, ModelTrainer, TrainerConfig, TrainingBatch,
                                                    TrainingVariables)
from nnabla_rl.models import Model, Perturbator, QFunction, VariationalAutoEncoder
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


@dataclass
class BCQPerturbatorTrainerConfig(TrainerConfig):
    '''
    Args:
        phi(float): action perturbator noise coefficient
    '''
    phi: float = 0.05


class BCQPerturbatorTrainer(ModelTrainer):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: BCQPerturbatorTrainerConfig
    _q_function: QFunction
    _vae: VariationalAutoEncoder
    _perturbator_loss: nn.Variable

    def __init__(self,
                 models: Union[Perturbator, Sequence[Perturbator]],
                 solvers: Dict[str, nn.solver.Solver],
                 q_function: QFunction,
                 vae: VariationalAutoEncoder,
                 env_info: EnvironmentInfo,
                 config: BCQPerturbatorTrainerConfig = BCQPerturbatorTrainerConfig()):
        self._q_function = q_function
        self._vae = vae
        super(BCQPerturbatorTrainer, self).__init__(models, solvers, env_info, config)

    def _update_model(self,
                      models: Sequence[Model],
                      solvers: Dict[str, nn.solver.Solver],
                      batch: TrainingBatch,
                      training_variables: TrainingVariables,
                      **kwargs) -> Dict:
        for t, b in zip(training_variables, batch):
            set_data_to_variable(t.s_current, b.s_current)

        # update model
        for solver in solvers.values():
            solver.zero_grad()
        self._perturbator_loss.forward(clear_no_need_grad=True)
        self._perturbator_loss.backward(clear_buffer=True)
        for solver in solvers.values():
            solver.update()

        trainer_state = {}
        trainer_state['perturbator_loss'] = float(self._perturbator_loss.d.copy())
        return trainer_state

    def _build_training_graph(self, models: Union[Model, Sequence[Model]], training_variables: TrainingVariables):
        self._perturbator_loss = 0
        models = cast(Sequence[Perturbator], models)
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
        assert training_variables.s_current is not None
        batch_size = training_variables.batch_size

        models = cast(Sequence[Perturbator], models)
        for perturbator in models:
            action = self._vae.decode(z=None, state=training_variables.s_current)
            action.need_grad = False

            noise = perturbator.generate_noise(training_variables.s_current, action, phi=self._config.phi)

            xi_loss = -self._q_function.q(training_variables.s_current, action + noise)
            assert xi_loss.shape == (batch_size, 1)

            self._perturbator_loss += 0.0 if ignore_loss else NF.mean(xi_loss)

    def _setup_training_variables(self, batch_size) -> TrainingVariables:
        # Training input variables
        s_current_var = create_variable(batch_size, self._env_info.state_shape)
        return TrainingVariables(batch_size, s_current_var)

    @property
    def loss_variables(self) -> Dict[str, nn.Variable]:
        return {"perturbator_loss": self._perturbator_loss}
