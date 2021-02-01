import multiprocessing as mp
import numpy as np
import os
from dataclasses import dataclass
from collections import namedtuple
from typing import List, Optional

import nnabla as nn
from nnabla import functions as NF
from nnabla import solvers as NS

import nnabla_rl.utils.context as context
import nnabla_rl.functions as RF
from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.builders import VFunctionBuilder, StochasticPolicyBuilder, SolverBuilder
from nnabla_rl import environment_explorers as EE
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.utils.data import marshall_experiences, unzip
from nnabla_rl.models import StochasticPolicy, VFunction, A3CSharedFunctionHead, A3CPolicy, A3CVFunction
from nnabla_rl.utils.multiprocess import (mp_to_np_array, np_to_mp_array,
                                          mp_array_from_np_array, new_mp_arrays_from_params,
                                          copy_mp_arrays_to_params, copy_params_to_mp_arrays)
from nnabla_rl.utils.reproductions import set_global_seed


@dataclass
class A2CParam(AlgorithmParam):
    gamma: float = 0.99
    n_steps: int = 5
    learning_rate: float = 7e-4
    v_function_coef: float = 0.5
    entropy_coef: float = 0.01
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
        if not ((0.0 <= self.gamma) & (self.gamma <= 1.0)):
            raise ValueError('gamma must lie between [0.0, 1.0]')
        self._assert_positive(self.n_steps, 'n_steps')
        self._assert_positive(self.learning_rate, 'learning_rate')
        self._assert_positive(self.max_grad_norm, 'max_grad_norm')


class DefaultPolicyBuilder(StochasticPolicyBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: A2CParam,
                    **kwargs) -> StochasticPolicy:
        _shared_function_head = A3CSharedFunctionHead(scope_name="shared",
                                                      state_shape=env_info.state_shape)
        return A3CPolicy(head=_shared_function_head,
                         scope_name="shared",
                         state_shape=env_info.state_shape,
                         action_dim=env_info.action_dim)


class DefaultVFunctionBuilder(VFunctionBuilder):
    def build_model(self,  # type: ignore[override]
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: A2CParam,
                    **kwargs) -> VFunction:
        _shared_function_head = A3CSharedFunctionHead(scope_name="shared",
                                                      state_shape=env_info.state_shape)
        return A3CVFunction(head=_shared_function_head,
                            scope_name="shared",
                            state_shape=env_info.state_shape)


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,  # type: ignore[override]
                     env_info: EnvironmentInfo,
                     algorithm_params: A2CParam,
                     **kwargs) -> nn.solver.Solver:
        return NS.RMSprop(lr=algorithm_params.learning_rate,
                          decay=algorithm_params.decay,
                          eps=algorithm_params.epsilon)


class A2C(Algorithm):
    _params: A2CParam
    _gpu_id: int
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
    _policy_loss: nn.Variable
    _v_function_loss: nn.Variable

    def __init__(self, env_or_env_info,
                 v_function_builder: VFunctionBuilder = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: StochasticPolicyBuilder = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 params=A2CParam()):
        self._gpu_id = context._gpu_id
        # Prevent setting context by the Algorithm class
        if 0 <= self._gpu_id:
            context._gpu_id = -1
        super(A2C, self).__init__(env_or_env_info, params=params)

        self._v_function = v_function_builder('v', self._env_info, self._params)
        self._policy = policy_builder('pi', self._env_info, self._params)

        self._policy_solver = policy_solver_builder(self._env_info, self._params)
        self._v_function_solver = v_solver_builder(self._env_info, self._params)

    @eval_api
    def compute_eval_action(self, state):
        return self._compute_action(state)

    def _compute_action(self, s):
        s = np.expand_dims(s, axis=0)
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
        context._gpu_id = self._gpu_id

        self._setup_policy_training(env_or_buffer)
        self._setup_v_function_training(env_or_buffer)

        self._actors, self._actor_processes = self._launch_actor_processes(env)

        context._set_nnabla_context()

        self._build_training_graph(env_or_buffer)
        self._setup_policy_training(env_or_buffer)
        self._setup_v_function_training(env_or_buffer)

    def _build_training_graph(self, env):
        n_steps = self._params.n_steps
        actor_num = self._params.actor_num
        batch_size = n_steps * actor_num

        state_shape = env.observation_space.shape
        self._s_current_var = nn.Variable([batch_size, *state_shape])
        if self._env_info.is_discrete_action_env():
            action_dim = 1
        else:
            action_dim = env.action_space.shape[0]
        self._a_current_var = nn.Variable([batch_size, action_dim])
        self._returns_var = nn.Variable([batch_size, 1])

        distribution = self._policy.pi(self._s_current_var)
        log_prob = distribution.log_prob(self._a_current_var)
        entropy = distribution.entropy()
        v_var = self._v_function.v(self._s_current_var)

        advantage = self._returns_var - v_var
        advantage.need_grad = False

        pi_loss = NF.mean(-advantage * log_prob - self._params.entropy_coef * entropy)
        v_loss = self._params.v_function_coef * RF.mean_squared_error(self._returns_var, v_var)

        self._policy_loss = pi_loss
        self._v_function_loss = v_loss

    def _setup_policy_training(self, env_or_buffer):
        dummy_state = env_or_buffer.observation_space.sample()
        dummy_state = nn.Variable.from_numpy_array(np.expand_dims(dummy_state, axis=0))

        self._policy.pi(dummy_state)
        self._policy_solver.set_parameters(self._policy.get_parameters())
        return None

    def _setup_v_function_training(self, env_or_buffer):
        dummy_state = env_or_buffer.observation_space.sample()
        dummy_state = nn.Variable.from_numpy_array(np.expand_dims(dummy_state, axis=0))

        self._v_function.v(dummy_state)
        self._v_function_solver.set_parameters(self._v_function.get_parameters())
        return None

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
        for i in range(self._params.actor_num):
            actor = _A2CActor(actor_num=i,
                              env=env,
                              env_info=self._env_info,
                              v_function=v_function,
                              policy=policy,
                              params=self._params)
            actors.append(actor)
        return actors

    def _kill_actor_processes(self, process):
        process.terminate()
        process.join()

    def _run_online_training_iteration(self, env):
        update_interval = self._params.n_steps * self._params.actor_num
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
        # lr decay
        alpha = self._params.learning_rate * (1.0 - self._iteration_num / self.max_iterations)
        self._policy_solver.set_learning_rate(alpha)
        self._v_function_solver.set_learning_rate(alpha)

        s, a, returns = experiences
        self._s_current_var.d = s
        self._a_current_var.d = a
        self._returns_var.d = returns

        # model update
        self._policy_solver.zero_grad()
        self._policy_loss.forward(clear_no_need_grad=True)
        self._policy_loss.backward(clear_buffer=True)
        if self._params.max_grad_norm is not None:
            self._clip_grad_by_global_norm(self._policy_solver, clip_norm=self._params.max_grad_norm)
        self._policy_solver.update()

        self._v_function_solver.zero_grad()
        self._v_function_loss.forward(clear_no_need_grad=True)
        self._v_function_loss.backward(clear_buffer=True)
        if self._params.max_grad_norm is not None:
            self._clip_grad_by_global_norm(self._v_function_solver, clip_norm=self._params.max_grad_norm)
        self._v_function_solver.update()

    def _clip_grad_by_global_norm(self, solver, clip_norm):
        parameters = solver.get_parameters()
        global_norm = np.linalg.norm([np.linalg.norm(param.g) for param in parameters.values()])
        scalar = clip_norm / global_norm
        if scalar < 1.0:
            solver.scale_grad(scalar)

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


class _A2CActor(object):
    def __init__(self, actor_num, env, env_info, policy, v_function, params):
        self._actor_num = actor_num
        self._env = env
        self._env_info = env_info
        self._policy = policy
        self._v_function = v_function
        self._n_steps = params.n_steps
        self._gamma = params.gamma
        self._params = params

        # IPC communication variables
        self._disposed = mp.Value('i', False)
        self._task_start_event = mp.Event()
        self._task_finish_event = mp.Event()

        self._policy_mp_arrays = new_mp_arrays_from_params(policy.get_parameters())
        self._v_function_mp_arrays = new_mp_arrays_from_params(v_function.get_parameters())

        explorer_params = EE.RawPolicyExplorerParam(initial_step_num=0,
                                                    timelimit_as_terminal=self._params.timelimit_as_terminal)
        self._environment_explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                                          env_info=self._env_info,
                                                          params=explorer_params)

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
        context._set_nnabla_context()
        if self._params.seed >= 0:
            seed = self._actor_num + self._params.seed
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
        (s, a, r, non_terminal) = marshall_experiences(experiences)
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
