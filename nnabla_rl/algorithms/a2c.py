# Copyright 2021 Sony Corporation.
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

import multiprocessing as mp
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import nnabla as nn
import nnabla_rl.model_trainers as MT
import nnabla_rl.utils.context as context
from nnabla import solvers as NS
from nnabla_rl import environment_explorers as EE
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import ModelTrainer, TrainingBatch
from nnabla_rl.models import A3CPolicy, A3CSharedFunctionHead, A3CVFunction, StochasticPolicy, VFunction
from nnabla_rl.utils.data import marshal_experiences, unzip
from nnabla_rl.utils.multiprocess import (copy_mp_arrays_to_params, copy_params_to_mp_arrays, mp_array_from_np_array,
                                          mp_to_np_array, new_mp_arrays_from_params, np_to_mp_array)
from nnabla_rl.utils.reproductions import set_global_seed


@dataclass
class A2CConfig(AlgorithmConfig):
    """
    List of configurations for A2C algorithm

    Args:
        gamma (float): discount factor of rewards. Defaults to 0.99.
        n_steps (int): number of rollout steps. Defaults to 5.
        learning_rate (float): learning rate which is set to all solvers. \
            You can customize/override the learning rate for each solver by implementing the \
            (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`) by yourself. \
            Defaults to 0.0007.
        entropy_coefficient (float): scalar of entropy regularization term. Defaults to 0.01.
        value_coefficient (float): scalar of value loss. Defaults to 0.5.
        decay (float): decay parameter of Adam solver. Defaults to 0.99.
        epsilon (float): epislon of Adam solver. Defaults to 0.00001.
        start_timesteps (int): the timestep when training starts.\
            The algorithm will collect experiences from the environment by acting randomly until this timestep.
            Defaults to 1.
        actor_num (int): number of parallel actors. Defaults to 8.
        timelimit_as_terminal (bool): Treat as done if the environment reaches the \
            `timelimit <https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py>`_.\
            Defaults to False.
        max_grad_norm (float): threshold value for clipping gradient. Defaults to 0.5.
        seed (int): base seed of random number generator used by the actors. Defaults to 1.
    """
    gamma: float = 0.99
    n_steps: int = 5
    learning_rate: float = 7e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    decay: float = 0.99
    epsilon: float = 1e-5
    start_timesteps: int = 1
    actor_num: int = 8
    timelimit_as_terminal: bool = False
    max_grad_norm: Optional[float] = 0.5
    seed: int = -1

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_between(self.decay, 0.0, 1.0, 'decay')
        self._assert_positive(self.n_steps, 'n_steps')
        self._assert_positive(self.actor_num, 'actor num')
        self._assert_positive(self.learning_rate, 'learning_rate')


class DefaultPolicyBuilder(ModelBuilder[StochasticPolicy]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: A2CConfig,
                    **kwargs) -> StochasticPolicy:
        _shared_function_head = A3CSharedFunctionHead(scope_name="shared",
                                                      state_shape=env_info.state_shape)
        return A3CPolicy(head=_shared_function_head,
                         scope_name="shared",
                         state_shape=env_info.state_shape,
                         action_dim=env_info.action_dim)


class DefaultVFunctionBuilder(ModelBuilder[VFunction]):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_config: A2CConfig,
                    **kwargs) -> VFunction:
        _shared_function_head = A3CSharedFunctionHead(scope_name="shared",
                                                      state_shape=env_info.state_shape)
        return A3CVFunction(head=_shared_function_head,
                            scope_name="shared",
                            state_shape=env_info.state_shape)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_config: A2CConfig,
                     **kwargs) -> nn.solver.Solver:
        return NS.RMSprop(lr=algorithm_config.learning_rate,
                          decay=algorithm_config.decay,
                          eps=algorithm_config.epsilon)


class A2C(Algorithm):
    '''Advantage Actor-Critic (A2C) algorithm implementation.

    This class implements the Advantage Actor-Critic (A2C) algorithm.
    A2C is the synchronous version of A3C, Asynchronous Advantage Actor-Critic.
    A3C was proposed by V. Mnih, et al. in the paper: "Asynchronous Methods for Deep Reinforcement Learning"
    For detail see: https://arxiv.org/abs/1602.01783

    This algorithm only supports online training.

    Args:
        env_or_env_info\
        (gym.Env or :py:class:`EnvironmentInfo <nnabla_rl.environments.environment_info.EnvironmentInfo>`):
            the environment to train or environment info
        v_function_builder (:py:class:`ModelBuilder[VFunction] <nnabla_rl.builders.ModelBuilder>`):
            builder of v function models
        v_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for v function solvers
        policy_builder (:py:class:`ModelBuilder[StochasicPolicy] <nnabla_rl.builders.ModelBuilder>`):
            builder of policy models
        policy_solver_builder (:py:class:`SolverBuilder <nnabla_rl.builders.SolverBuilder>`): builder for policy solvers
        config (:py:class:`A2CConfig <nnabla_rl.algorithms.a2c.A2CConfig>`): configuration of A2C algorithm
    '''

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: A2CConfig
    _v_function: VFunction
    _v_function_solver: nn.solver.Solver
    _policy: StochasticPolicy
    _policy_solver: nn.solver.Solver
    _actors: List['_A2CActor']
    _actor_processes: List[mp.Process]
    _eval_state_var: nn.Variable
    _eval_action: nn.Variable
    _s_current_var: nn.Variable
    _a_current_var: nn.Variable
    _returns_var: nn.Variable

    _policy_trainer: ModelTrainer
    _v_function_trainer: ModelTrainer

    _policy_solver_builder: SolverBuilder
    _v_solver_builder: SolverBuilder

    _policy_trainer_state: Dict[str, Any]
    _v_function_trainer_state: Dict[str, Any]

    def __init__(self, env_or_env_info,
                 v_function_builder: ModelBuilder[VFunction] = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder[StochasticPolicy] = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 config=A2CConfig()):
        super(A2C, self).__init__(env_or_env_info, config=config)
        if self._env_info.is_continuous_action_env():
            raise NotImplementedError

        # Initialize on cpu and change the context later
        with nn.context_scope(context.get_nnabla_context(-1)):
            self._policy = policy_builder('pi', self._env_info, self._config)
            self._v_function = v_function_builder('v', self._env_info, self._config)

            self._policy_solver = policy_solver_builder(self._env_info, self._config)
            self._policy_solver_builder = policy_solver_builder  # keep for later use
            self._v_function_solver = v_solver_builder(self._env_info, self._config)
            self._v_solver_builder = v_solver_builder  # keep for later use

    @eval_api
    def compute_eval_action(self, state):
        with nn.context_scope(context.get_nnabla_context(self._config.gpu_id)):
            s = np.expand_dims(state, axis=0)
            if not hasattr(self, '_eval_state_var'):
                self._eval_state_var = nn.Variable(s.shape)
                distribution = self._policy.pi(self._eval_state_var)
                self._eval_action = distribution.sample()
                self._eval_action.need_grad = False
            self._eval_state_var.d = s
            self._eval_action.forward(clear_no_need_grad=True)
            action = np.squeeze(self._eval_action.d, axis=0)
            if self._env_info.is_discrete_action_env():
                return np.int(action)
            else:
                return action

    def _before_training_start(self, env_or_buffer):
        if not self._is_env(env_or_buffer):
            raise ValueError('A2C only supports online training')
        env = env_or_buffer

        # FIXME: This setup is a workaround for creating underlying model parameters
        # If the parameter is not created, the multiprocessable array (created in launch_actor_processes)
        # will be empty and the agent does not learn anything
        context.set_nnabla_context(-1)
        self._setup_policy_training(env)
        self._setup_v_function_training(env)

        self._actors, self._actor_processes = self._launch_actor_processes(env)

        # NOTE: Setting gpu context after the launch of processes
        # If you set the gpu context before the launch of proceses, the process will corrupt
        context.set_nnabla_context(self._config.gpu_id)

        # Setup again here to use gpu (if it is set)
        old_policy_solver = self._policy_solver
        self._policy_solver = self._policy_solver_builder(self._env_info, self._config)
        self._policy_trainer = self._setup_policy_training(env)
        self._policy_solver.set_states(old_policy_solver.get_states())

        old_v_function_solver = self._v_function_solver
        self._v_function_solver = self._v_solver_builder(self._env_info, self._config)
        self._v_function_trainer = self._setup_v_function_training(env)
        self._v_function_solver.set_states(old_v_function_solver.get_states())

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_config = MT.policy_trainers.A2CPolicyTrainerConfig(
            entropy_coefficient=self._config.entropy_coefficient,
            max_grad_norm=self._config.max_grad_norm
        )
        policy_trainer = MT.policy_trainers.A2CPolicyTrainer(
            models=self._policy,
            solvers={self._policy.scope_name: self._policy_solver},
            env_info=self._env_info,
            config=policy_trainer_config)
        return policy_trainer

    def _setup_v_function_training(self, env_or_buffer):
        # training input/loss variables
        v_function_trainer_config = MT.v_value.MonteCarloVTrainerConfig(
            reduction_method='mean',
            v_loss_scalar=self._config.value_coefficient,
            max_grad_norm=self._config.max_grad_norm
        )
        v_function_trainer = MT.v_value.MonteCarloVTrainer(
            train_functions=self._v_function,
            solvers={self._v_function.scope_name: self._v_function_solver},
            env_info=self._env_info,
            config=v_function_trainer_config
        )
        return v_function_trainer

    def _launch_actor_processes(self, env):
        actors = self._build_a2c_actors(env, v_function=self._v_function, policy=self._policy)
        processes = []
        for actor in actors:
            p = mp.Process(target=actor, daemon=True)
            p.start()
            processes.append(p)
        return actors, processes

    def _build_a2c_actors(self, env, v_function, policy):
        actors = []
        for i in range(self._config.actor_num):
            actor = _A2CActor(actor_num=i,
                              env=env,
                              env_info=self._env_info,
                              v_function=v_function,
                              policy=policy,
                              config=self._config)
            actors.append(actor)
        return actors

    def _after_training_finish(self, env_or_buffer):
        for actor in self._actors:
            actor.dispose()
        for process in self._actor_processes:
            self._kill_actor_processes(process)

    def _kill_actor_processes(self, process):
        process.terminate()
        process.join()

    def _run_online_training_iteration(self, env):
        update_interval = self._config.n_steps * self._config.actor_num
        if self.iteration_num % update_interval != 0:
            return
        experiences = self._collect_experiences(self._actors)
        self._a2c_training(experiences)

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _collect_experiences(self, actors):
        for actor in actors:
            actor.update_v_params(self._v_function.get_parameters())
            actor.update_policy_params(self._policy.get_parameters())

            actor.run_data_collection()

        results = [actor.wait_data_collection() for actor in actors]
        return (np.concatenate(item, axis=0) for item in unzip(results))

    def _a2c_training(self, experiences):
        s, a, returns = experiences
        advantage = self._compute_advantage(s, returns)
        extra = {}
        extra['advantage'] = advantage
        extra['v_target'] = returns
        batch = TrainingBatch(batch_size=len(a),
                              s_current=s,
                              a_current=a,
                              extra=extra)

        # lr decay
        alpha = self._config.learning_rate * (1.0 - self._iteration_num / self.max_iterations)
        self._policy_trainer.set_learning_rate(alpha)
        self._v_function_trainer.set_learning_rate(alpha)

        # model update
        self._policy_trainer_state = self._policy_trainer.train(batch)
        self._v_function_trainer_state = self._v_function_trainer.train(batch)

    def _compute_advantage(self, s, returns):
        if not hasattr(self, '_state_var_for_advantage'):
            self._state_var_for_advantage = nn.Variable(s.shape)
            self._returns_var_for_advantage = nn.Variable(returns.shape)
            v_for_advantage = self._v_function.v(self._state_var_for_advantage)
            self._advantage = self._returns_var_for_advantage - v_for_advantage
            self._advantage.need_grad = False

        self._state_var_for_advantage.d = s
        self._returns_var_for_advantage.d = returns
        self._advantage.forward(clear_no_need_grad=True)
        return self._advantage.d

    def _models(self):
        models = {}
        models[self._policy.scope_name] = self._policy
        models[self._v_function.scope_name] = self._v_function
        return models

    def _solvers(self):
        solvers = {}
        solvers[self._policy.scope_name] = self._policy_solver
        solvers[self._v_function.scope_name] = self._v_function_solver
        return solvers

    @property
    def latest_iteration_state(self):
        latest_iteration_state = super(A2C, self).latest_iteration_state
        if hasattr(self, '_policy_trainer_state'):
            latest_iteration_state['scalar'].update({'pi_loss': self._policy_trainer_state['pi_loss']})
        if hasattr(self, '_v_function_trainer_state'):
            latest_iteration_state['scalar'].update({'v_loss': self._v_function_trainer_state['v_loss']})
        return latest_iteration_state


class _A2CActor(object):
    def __init__(self, actor_num, env, env_info, policy, v_function, config):
        self._actor_num = actor_num
        self._env = env
        self._env_info = env_info
        self._policy = policy
        self._v_function = v_function
        self._n_steps = config.n_steps
        self._gamma = config.gamma
        self._config = config

        # IPC communication variables
        self._disposed = mp.Value('i', False)
        self._task_start_event = mp.Event()
        self._task_finish_event = mp.Event()

        self._policy_mp_arrays = new_mp_arrays_from_params(policy.get_parameters())
        self._v_function_mp_arrays = new_mp_arrays_from_params(v_function.get_parameters())

        explorer_config = EE.RawPolicyExplorerConfig(initial_step_num=0,
                                                     timelimit_as_terminal=self._config.timelimit_as_terminal)
        self._environment_explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                                          env_info=self._env_info,
                                                          config=explorer_config)

        obs_space = self._env.observation_space
        action_space = self._env.action_space

        MultiProcessingArrays = namedtuple('MultiProcessingArrays', ['state', 'action', 'returns'])

        state_mp_array_shape = (self._n_steps, *obs_space.shape)
        state_mp_array = mp_array_from_np_array(
            np.empty(shape=state_mp_array_shape, dtype=obs_space.dtype))
        if env_info.is_discrete_action_env():
            action_mp_array_shape = (self._n_steps, 1)
            action_mp_array = mp_array_from_np_array(
                np.empty(shape=action_mp_array_shape, dtype=action_space.dtype))
        else:
            action_mp_array_shape = (self._n_steps, action_space.shape[0])
            action_mp_array = mp_array_from_np_array(
                np.empty(shape=action_mp_array_shape, dtype=action_space.dtype))

        scalar_mp_array_shape = (self._n_steps, 1)
        returns_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))

        self._mp_arrays = MultiProcessingArrays(
            (state_mp_array, state_mp_array_shape, obs_space.dtype),
            (action_mp_array, action_mp_array_shape, action_space.dtype),
            (returns_mp_array, scalar_mp_array_shape, np.float32)
        )

    def __call__(self):
        self._run_actor_loop()

    def dispose(self):
        self._disposed = True
        self._task_start_event.set()

    def run_data_collection(self):
        self._task_finish_event.clear()
        self._task_start_event.set()

    def wait_data_collection(self):
        self._task_finish_event.wait()
        return (mp_to_np_array(mp_array, shape, dtype) for (mp_array, shape, dtype) in self._mp_arrays)

    def update_v_params(self, params):
        self._update_params(src=params, dest=self._v_function_mp_arrays)

    def update_policy_params(self, params):
        self._update_params(src=params, dest=self._policy_mp_arrays)

    def _run_actor_loop(self):
        context.set_nnabla_context(self._config.gpu_id)
        if self._config.seed >= 0:
            seed = self._actor_num + self._config.seed
        else:
            seed = os.getpid()
        set_global_seed(seed)
        self._env.seed(seed)
        while (True):
            self._task_start_event.wait()
            if self._disposed.get_obj():
                break
            self._synchronize_policy_params(self._policy.get_parameters())
            self._synchronize_v_function_params(self._v_function.get_parameters())

            experiences = self._run_data_collection()
            self._fill_result(experiences)

            self._task_start_event.clear()
            self._task_finish_event.set()

    def _run_data_collection(self):
        experiences = self._environment_explorer.step(self._env, n=self._n_steps, break_if_done=False)
        s_last = experiences[-1][4]
        experiences = [(s, a, r, non_terminal)
                       for (s, a, r, non_terminal, *_) in experiences]
        processed_experiences = self._process_experiences(experiences, s_last)
        return processed_experiences

    def _process_experiences(self, experiences, s_last):
        (s, a, r, non_terminal) = marshal_experiences(experiences)
        v_last = self._compute_v(s_last)
        returns = self._compute_returns(r, non_terminal, v_last)
        return (s, a, returns)

    def _compute_returns(self, rewards, non_terminals, value_last):
        returns = []
        R = value_last
        for i, (r, non_terminal) in enumerate(zip(rewards[::-1], non_terminals[::-1])):
            R = r + self._gamma * R * non_terminal
            returns.insert(0, [R])
        return np.array(returns)

    def _compute_v(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_state_var'):
            self._state_var = nn.Variable(s.shape)
            self._v_var = self._v_function.v(self._state_var)
            self._v_var.need_grad = False
        self._state_var.d = s
        self._v_var.forward(clear_no_need_grad=True)
        v = self._v_var.d.copy()
        return v

    def _fill_result(self, experiences):
        def array_and_dtype(mp_arrays_item):
            return mp_arrays_item[0], mp_arrays_item[2]
        (s, a, returns) = experiences
        np_to_mp_array(s, *array_and_dtype(self._mp_arrays.state))
        np_to_mp_array(a, *array_and_dtype(self._mp_arrays.action))
        np_to_mp_array(returns, *array_and_dtype(self._mp_arrays.returns))

    @eval_api
    def _compute_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._policy.pi(self._eval_state_var)
            self._eval_action = distribution.sample()
            self._eval_state_var.need_grad = False
            self._eval_action.need_grad = False
        self._eval_state_var.d = s
        self._eval_action.forward(clear_no_need_grad=True)
        action = np.squeeze(self._eval_action.d, axis=0)
        if self._env_info.is_discrete_action_env():
            return np.int(action), {}
        else:
            return action, {}

    def _update_params(self, src, dest):
        copy_params_to_mp_arrays(src, dest)

    def _synchronize_policy_params(self, params):
        self._synchronize_params(src=self._policy_mp_arrays, dest=params)

    def _synchronize_v_function_params(self, params):
        self._synchronize_params(src=self._v_function_mp_arrays, dest=params)

    def _synchronize_params(self, src, dest):
        copy_mp_arrays_to_params(src, dest)
