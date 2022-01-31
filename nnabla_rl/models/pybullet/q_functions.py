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

from typing import Optional, Tuple

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.functions as RF
from nnabla_rl.models.q_function import ContinuousQFunction
from nnabla_rl.scopes import is_eval_scope


class ICRA2018QtOptQFunction(ContinuousQFunction):
    '''
    Q function proposed in paper for grasping environment.
    See: https://arxiv.org/pdf/1802.10264.pdf
    '''

    def __init__(
        self,
        scope_name: str,
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        cem_initial_mean: Optional[Tuple[float, ...]] = None,
        cem_initial_variance: Optional[Tuple[float, ...]] = None,
        cem_sample_size: int = 500,
        cem_num_elites: int = 50,
        cem_num_iterations: int = 100,
        cem_alpha: float = 0.0,
        random_sample_size: int = 500
    ):
        super(ICRA2018QtOptQFunction, self).__init__(scope_name)
        self._action_high = action_high
        self._action_low = action_low

        self._cem_initial_mean_numpy = np.zeros(action_dim) if cem_initial_mean is None else np.array(cem_initial_mean)
        self._cem_initial_variance_numpy = 0.5 * \
            np.ones(action_dim) if cem_initial_variance is None else np.array(cem_initial_variance)
        self._cem_sample_size = cem_sample_size
        self._cem_num_elites = cem_num_elites
        self._cem_num_iterations = cem_num_iterations
        self._cem_alpha = cem_alpha

        self._random_sample_size = random_sample_size

    def q(self, s: Tuple[nn.Variable, nn.Variable], a: nn.Variable) -> nn.Variable:
        image, timestep = s
        # timestep.shape = (batch_size, 1)
        batch_size = timestep.shape[0]
        tiled_time_step = NF.tile(timestep, 49)
        # timestep.shape = (batch_size, 1, 7, 7)
        tiled_time_step = NF.reshape(tiled_time_step, (batch_size, 1, 7, 7))

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope('state_conv1'):
                h = NF.relu(NPF.convolution(image, 32, (3, 3), stride=(2, 2)))

            with nn.parameter_scope('state_conv2'):
                h = NF.relu(NPF.convolution(h, 32, (3, 3), stride=(2, 2)))

            with nn.parameter_scope('state_conv3'):
                h = NF.relu(NPF.convolution(h, 32, (3, 3), stride=(2, 2)))

            encoded_state = NF.concatenate(tiled_time_step, h, axis=1)

            with nn.parameter_scope('action_affine1'):
                h = NF.relu(NPF.affine(a, 33))
                encoded_action = NF.reshape(h, (batch_size, 33, 1, 1))

            h = encoded_state + encoded_action
            h = NF.reshape(h, (batch_size, -1))

            with nn.parameter_scope('affine1'):
                h = NF.relu(NPF.affine(h, 32))

            with nn.parameter_scope('affine2'):
                h = NF.relu(NPF.affine(h, 32))

            with nn.parameter_scope('affine3'):
                q_value = NPF.affine(h, 1)

        return q_value

    def max_q(self, s: Tuple[nn.Variable, nn.Variable]) -> nn.Variable:
        return self.q(s, self.argmax_q(s))

    def argmax_q(self, s: Tuple[nn.Variable, nn.Variable]) -> nn.Variable:
        tile_size = self._cem_sample_size if is_eval_scope() else self._random_sample_size
        tiled_s = tuple([self._tile_state(each_s, tile_size) for each_s in s])
        batch_size = s[0].shape[0]

        def objective_function(a: nn.Variable) -> nn.Variable:
            batch_size, sample_size, action_dim = a.shape
            a = a.reshape((batch_size*sample_size, action_dim))
            q_value = self.q(tiled_s, a)  # type: ignore
            q_value = q_value.reshape((batch_size, sample_size, 1))
            return q_value

        if is_eval_scope():
            initial_mean_var = nn.Variable.from_numpy_array(np.tile(self._cem_initial_mean_numpy, (batch_size, 1)))
            initial_variance_var = nn.Variable.from_numpy_array(
                np.tile(self._cem_initial_variance_numpy, (batch_size, 1)))
            optimized_action, _ = RF.gaussian_cross_entropy_method(
                objective_function,
                initial_mean_var,
                initial_variance_var,
                sample_size=self._cem_sample_size,
                num_elites=self._cem_num_elites,
                num_iterations=self._cem_num_iterations,
                alpha=self._cem_alpha
            )
        else:
            upper_bound = np.tile(self._action_high, (batch_size, 1))
            lower_bound = np.tile(self._action_low, (batch_size, 1))
            optimized_action = RF.random_shooting_method(
                objective_function,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                sample_size=self._random_sample_size
            )

        return optimized_action

    def _tile_state(self, s: nn.Variable, tile_size: int) -> nn.Variable:
        tile_reps = [tile_size, ] + [1, ] * len(s.shape)
        s = NF.tile(s, tile_reps)
        transpose_reps = [1, 0, ] + list(range(len(s.shape)))[2:]
        s = NF.transpose(s, transpose_reps)
        s = NF.reshape(s, (-1, *s.shape[2:]))
        return s
