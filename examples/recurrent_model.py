# Copyright 2021,2022,2023,2024 Sony Group Corporation.
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

from typing import Dict, Optional, Tuple

import gym

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla.solvers as NS
import nnabla_rl.algorithms as A
import nnabla_rl.hooks as H
import nnabla_rl.writers as W
from nnabla_rl.algorithm import AlgorithmConfig
from nnabla_rl.builders.model_builder import ModelBuilder
from nnabla_rl.builders.solver_builder import SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.environments.wrappers import NumpyFloat32Env, ScreenRenderEnv
from nnabla_rl.models.q_function import DiscreteQFunction, QFunction


class QFunctionWithRNN(DiscreteQFunction):
    # Define model (with RNN layers) to train
    def __init__(self, scope_name, n_action):
        super().__init__(scope_name)
        self._h = None
        self._c = None
        self._lstm_state_size = 50
        self._n_action = n_action

    def all_q(self, s: nn.Variable) -> nn.Variable:
        # Definition of q function
        with nn.parameter_scope(self.scope_name):
            h = NPF.affine(s, 50, name="affine-1")
            h = NF.relu(h)
            if not self._is_internal_state_created():
                # Create internal state self._h, self._c if it is not created
                batch_size = s.shape[0]
                self._create_internal_states(batch_size)
            # Keep hidden state and lstm cell for later use
            self._h, self._c = NPF.lstm_cell(h, self._h, self._c, self._lstm_state_size)
            h = self._h
            q = NPF.affine(h, self._n_action, name="pred-q")
        return q

    def is_recurrent(self) -> bool:
        # Return True because this model contains RNN layer
        return True

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        # Return internal rnn state shapes
        # In case of LSTM, there are two internal states.
        shapes: Dict[str, nn.Variable] = {}
        # You can use arbitral (but distinguishable) key as name
        # Use same key for same state
        shapes["my_lstm_h"] = (self._lstm_state_size,)
        shapes["my_lstm_c"] = (self._lstm_state_size,)
        return shapes

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        # Return current internal states
        states: Dict[str, nn.Variable] = {}
        # You can use arbitral (but distinguishable) key as name
        # Use same key for same state.
        states["my_lstm_h"] = self._h
        states["my_lstm_c"] = self._c
        return states

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        if states is None:
            # Set states to 0 if states are None.
            if self._h is not None:
                self._h.data.zero()
            if self._c is not None:
                self._c.data.zero()
        else:
            # Otherwise, set given states
            # Use the key defined in internal_state_shapes() for getting the states
            self._h = states["my_lstm_h"]
            self._c = states["my_lstm_c"]

    def _create_internal_states(self, batch_size):
        self._h = nn.Variable((batch_size, self._lstm_state_size))
        self._c = nn.Variable((batch_size, self._lstm_state_size))

        self._h.data.zero()
        self._c.data.zero()

    def _is_internal_state_created(self) -> bool:
        return self._h is not None and self._c is not None


class QFunctionWithRNNBuilder(ModelBuilder[QFunction]):
    def build_model(
        self, scope_name: str, env_info: EnvironmentInfo, algorithm_config: AlgorithmConfig, **kwargs
    ) -> QFunction:
        action_num = env_info.action_dim
        return QFunctionWithRNN(scope_name, action_num)


class AdamSolverBuilder(SolverBuilder):
    def build_solver(self, env_info: EnvironmentInfo, algorithm_config: AlgorithmConfig, **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_config.learning_rate)


def build_env(seed=None):
    env = gym.make("MountainCar-v0")
    env = NumpyFloat32Env(env)
    env = ScreenRenderEnv(env)
    env.seed(seed)
    return env


def main():
    # Setup training env
    train_env = build_env()

    # Setup RL algorithm that supports RNN layers.
    # Here, we use DRQN algorithm
    # For the list of algorithms that support RNN layers see
    # https://github.com/sony/nnabla-rl/tree/master/nnabla_rl/algorithms
    config = A.DRQNConfig(
        gpu_id=0,
        learning_rate=1e-2,
        gamma=0.9,
        learner_update_frequency=1,
        target_update_frequency=200,
        start_timesteps=200,
        replay_buffer_size=10000,
        max_explore_steps=10000,
        initial_epsilon=1.0,
        final_epsilon=0.001,
        test_epsilon=0.001,
        grad_clip=None,
        unroll_steps=2,
    )  # Unroll only for 2 timesteps for fast iteration. Because this is an example
    drqn = A.DRQN(
        train_env, config=config, q_func_builder=QFunctionWithRNNBuilder(), q_solver_builder=AdamSolverBuilder()
    )

    # Optional: Add hooks to check the training progress
    eval_env = build_env(seed=100)
    evaluation_hook = H.EvaluationHook(
        eval_env,
        timing=1000,
        writer=W.FileWriter(outdir="./mountain_car_v0_drqn_results", file_prefix="evaluation_result"),
    )
    iteration_num_hook = H.IterationNumHook(timing=100)
    drqn.set_hooks(hooks=[iteration_num_hook, evaluation_hook])

    # Start the training
    drqn.train(train_env, total_iterations=50000)

    train_env.close()


if __name__ == "__main__":
    main()
