# Copyright 2021 Sony Corporation.
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

from typing import TYPE_CHECKING, Dict, Tuple, Union, cast

import numpy as np

import nnabla as nn
from nnabla.solver import Solver

if TYPE_CHECKING:
    from nnabla_rl.model_trainers.model_trainer import TrainingVariables

from nnabla_rl.models import Model
from nnabla_rl.typing import Shape


def sync_model(src: Model, dst: Model, tau: float = 1.0):
    copy_network_parameters(origin_params=src.get_parameters(), target_params=dst.get_parameters(), tau=tau)


def copy_network_parameters(origin_params, target_params, tau=1.0):
    if not ((0.0 <= tau) & (tau <= 1.0)):
        raise ValueError('tau must lie between [0.0, 1.0]')

    for key in target_params.keys():
        target_params[key].data.copy_from(origin_params[key].data * tau + target_params[key].data * (1 - tau))


def clip_grad_by_global_norm(solver: Solver, max_grad_norm: float):
    parameters = solver.get_parameters()
    global_norm = np.linalg.norm([np.linalg.norm(param.g) for param in parameters.values()])
    scalar = max_grad_norm / global_norm
    if scalar < 1.0:
        solver.scale_grad(scalar)


def create_variable(batch_size: int, shape: Shape) -> Union[nn.Variable, Tuple[nn.Variable, ...]]:
    if isinstance(shape, int):
        return nn.Variable((batch_size, shape))
    elif isinstance(shape[0], int):
        return nn.Variable((batch_size, *shape))
    else:
        shape = cast(Tuple[Tuple[int, ...], ...], shape)
        return tuple(nn.Variable((batch_size, *_shape)) for _shape in shape)


def create_variables(batch_size: int, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, nn.Variable]:
    variables: Dict[str, nn.Variale] = {}
    for name, shape in shapes.items():
        state: nn.Variable = create_variable(batch_size, shape)
        state.data.zero()
        variables[name] = state
    return variables


def retrieve_internal_states(scope_name: str,
                             prev_rnn_states: Dict[str, Dict[str, nn.Variable]],
                             train_rnn_states: Dict[str, Dict[str, nn.Variable]],
                             training_variables: 'TrainingVariables',
                             reset_on_terminal: bool) -> Dict[str, nn.Variable]:
    internal_states: Dict[str, nn.Variable] = {}
    if training_variables.is_initial_step():
        internal_states = train_rnn_states[scope_name]
    else:
        prev_states = prev_rnn_states[scope_name]
        train_states = train_rnn_states[scope_name]
        for state_name in train_states.keys():
            prev_state = prev_states[state_name]
            train_state = train_states[state_name]
            if reset_on_terminal:
                assert training_variables.prev_step_variables
                prev_non_terminal = training_variables.prev_step_variables.non_terminal
                internal_states[state_name] = prev_non_terminal * prev_state + (1.0 - prev_non_terminal) * train_state
            else:
                internal_states[state_name] = prev_state
    return internal_states
