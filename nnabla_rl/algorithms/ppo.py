import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S

from dataclasses import dataclass
from collections import namedtuple

import numpy as np

import multiprocessing as mp
from multiprocessing import sharedctypes

import os
import gym

import nnabla_rl
import nnabla_rl.models as M
import nnabla_rl.preprocessors as RP
import nnabla_rl.functions as RF
from nnabla_rl.algorithm import Algorithm, AlgorithmParam
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers import BufferIterator
from nnabla_rl.utils.data import marshall_experiences, unzip
from nnabla_rl.utils.multiprocess import mp_to_np_array, np_to_mp_array, mp_array_from_np_array, new_mp_arrays_from_params, copy_mp_arrays_to_params, copy_params_to_mp_arrays
import nnabla_rl.utils.context as context


def build_shared_policy(head, scope_name, state_shape, action_dim):
    return M.PPOAtariPolicy(head=head, scope_name=scope_name, state_shape=state_shape, action_dim=action_dim)


def build_shared_v_function(head, scope_name, state_shape):
    return M.PPOAtariVFunction(head=head, scope_name=scope_name, state_shape=state_shape)


def build_mujoco_policy(scope_name, state_shape, action_dim):
    return M.PPOMujocoPolicy(scope_name=scope_name, state_dim=state_shape, action_dim=action_dim)


def build_mujoco_v_function(scope_name, state_shape):
    return M.PPOMujocoVFunction(scope_name=scope_name, state_shape=state_shape)


def build_mujoco_state_preprocessor(scope_name, state_shape):
    return RP.RunningMeanNormalizer(scope_name, state_shape, value_clip=(-5.0, 5.0))


def build_discrete_env_policy_and_v_function(policy_builder, value_function_builder,
                                             state_shape, action_dim):
    shared_function_head = M.PPOSharedFunctionHead(
        scope_name="value_and_pi", state_shape=state_shape, action_dim=action_dim)
    if policy_builder is None:
        policy = build_shared_policy(head=shared_function_head,
                                     scope_name="value_and_pi",
                                     state_shape=state_shape,
                                     action_dim=action_dim)
    else:
        policy = policy_builder(
            scope_name="pi", state_shape=state_shape, action_dim=action_dim)
    if value_function_builder is None:
        v_function = build_shared_v_function(head=shared_function_head,
                                             scope_name="value_and_pi",
                                             state_shape=state_shape)
    else:
        v_function = value_function_builder(
            scope_name="v", state_shape=state_shape)
    return policy, v_function


def build_continuous_env_policy_and_v_function(policy_builder, value_function_builder,
                                               state_shape, action_dim):
    if policy_builder is None:
        policy = build_mujoco_policy(scope_name="pi",
                                     state_shape=state_shape,
                                     action_dim=action_dim)
    else:
        policy = policy_builder(
            scope_name="pi", state_shape=state_shape, action_dim=action_dim)
    if value_function_builder is None:
        v_function = build_mujoco_v_function(scope_name="v",
                                             state_shape=state_shape)
    else:
        v_function = value_function_builder(
            scope_name="v", state_shape=state_shape)
    return policy, v_function


def build_state_preprocessor(preprocessor_builder, state_shape):
    if preprocessor_builder is None:
        return build_mujoco_state_preprocessor(scope_name='preprocessor', state_shape=state_shape)
    else:
        return preprocessor_builder(scope_name='preprocessor', state_shape=state_shape)


@dataclass
class PPOParam(AlgorithmParam):
    gamma: float = 0.99
    learning_rate: float = 2.5*1e-4
    lmb: float = 0.95
    epsilon: float = 0.1
    entropy_coefficient: float = 0.01
    value_coefficient: float = 1.0
    actor_num: int = 8
    epochs: int = 3
    batch_size: int = 32 * 8
    actor_timesteps: int = 128
    total_timesteps: int = 10000
    decrease_alpha: bool = True
    only_reset_if_truncated: bool = True

    def __post_init__(self):
        '''__post_init__

        Check the set values are in valid range.

        '''
        self._assert_between(self.gamma, 0.0, 1.0, 'gamma')
        self._assert_positive(self.actor_num, 'actor num')
        self._assert_positive(self.epochs, 'epochs')
        self._assert_positive(self.batch_size, 'batch_size')
        self._assert_positive(self.actor_timesteps, 'actor_timesteps')
        self._assert_positive(self.total_timesteps, 'total_timesteps')


class PPO(Algorithm):
    '''Proximal Policy Optimization (PPO) algorithm implementation.

    This class implements the Proximal Policy Optimization (PPO) algorithm
    proposed by J. Schulman, et al. in the paper: "Proximal Policy Optimization Algorithms"
    For detail see: https://arxiv.org/pdf/1707.06347.pdf

    This algorithm only supports online training.
    '''

    def __init__(self, env_info,
                 value_function_builder=None,
                 policy_builder=None,
                 state_preprocessor_builder=None,
                 params=PPOParam()):
        self._gpu_id = context._gpu_id
        # Disable setting context by the Algorithm class
        if 0 <= self._gpu_id:
            context._gpu_id = -1
        super(PPO, self).__init__(env_info, params=params)

        state_shape = self._env_info.observation_space.shape
        if self._env_info.is_discrete_action_env():
            self._state_preprocessor = None
            action_dim = self._env_info.action_space.n
            self._policy, self._v_function = build_discrete_env_policy_and_v_function(
                policy_builder, value_function_builder, state_shape, action_dim)
        else:
            self._state_preprocessor = build_state_preprocessor(
                state_preprocessor_builder, state_shape)
            action_dim = self._env_info.action_space.shape[0]
            self._policy, self._v_function = build_continuous_env_policy_and_v_function(
                policy_builder, value_function_builder, state_shape[0], action_dim)
            self._policy.set_state_preprocessor(self._state_preprocessor)
            self._v_function.set_state_preprocessor(self._state_preprocessor)
        if self._state_preprocessor is not None:
            assert isinstance(self._state_preprocessor, RP.Preprocessor)
        assert isinstance(self._policy, M.StochasticPolicy)
        assert isinstance(self._v_function, M.VFunction)

        self._v_solver = None
        self._policy_solver = None

        self._state = None
        self._action = None
        self._next_state = None

        # training input/loss variables
        Variables = namedtuple('Variables',
                               ['s_current', 'a_current', 'log_prob', 'v_target', 'advantage', 'alpha'])
        if self._env_info.is_discrete_action_env():
            self._variables = Variables(
                nn.Variable((params.batch_size, *state_shape)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((1, 1)))
        else:
            self._variables = Variables(
                nn.Variable((params.batch_size, *state_shape)),
                nn.Variable((params.batch_size, action_dim)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((params.batch_size, 1)),
                nn.Variable((1, 1)))
        self._loss = None

        # evaluation input/action variables
        self._eval_state_var = nn.Variable((1, *state_shape))
        self._eval_action = None

        self._actors = None
        self._actor_processes = []

    def compute_eval_action(self, state):
        if context._gpu_id < 0 and 0 <= self._gpu_id:
            context._gpu_id = self._gpu_id
            context._set_nnabla_context()
        return self._compute_action(state)

    def _build_training_graph(self):
        distribution = self._policy.pi(self._variables.s_current)
        log_prob_new = distribution.log_prob(self._variables.a_current)
        log_prob_old = self._variables.log_prob
        probability_ratio = F.exp(log_prob_new - log_prob_old)
        clipped_ratio = F.clip_by_value(probability_ratio,
                                        1 - self._params.epsilon * self._variables.alpha,
                                        1 + self._params.epsilon * self._variables.alpha)
        lower_bounds = F.minimum2(probability_ratio * self._variables.advantage,
                                  clipped_ratio * self._variables.advantage)
        clip_loss = F.mean(lower_bounds)

        value = self._v_function.v(self._variables.s_current)
        value_loss = self._params.value_coefficient * \
            RF.mean_squared_error(value, self._variables.v_target)

        entropy = distribution.entropy()
        entropy_loss = F.mean(entropy)

        policy_loss = -clip_loss - self._params.entropy_coefficient * entropy_loss
        self._loss = value_loss + policy_loss

    def _build_evaluation_graph(self):
        distribution = self._policy.pi(self._eval_state_var)
        self._eval_action = distribution.sample()

    def _setup_solver(self):
        self._v_solver = S.Adam(self._params.learning_rate, eps=1e-5)
        self._v_solver.set_parameters(self._v_function.get_parameters())
        self._policy_solver = S.Adam(self._params.learning_rate, eps=1e-5)
        self._policy_solver.set_parameters(self._policy.get_parameters())

    def _before_training_start(self, env_or_buffer):
        if not self._is_env(env_or_buffer):
            raise ValueError('PPO only supports online training')
        context._gpu_id = self._gpu_id
        env = env_or_buffer
        self._actors, self._actor_processes = \
            self._launch_actor_processes(env)
        context._set_nnabla_context()
        # FIXME: Workaround to enable computing on gpu
        self._rebuild_computation_graph()

        old_v_solver = self._v_solver
        old_policy_solver = self._policy_solver
        self._setup_solver()
        self._v_solver.set_states(old_v_solver.get_states())
        self._policy_solver.set_states(old_policy_solver.get_states())

    def _after_training_finish(self, env_or_buffer):
        for actor in self._actors:
            actor.dispose()
        for process in self._actor_processes:
            self._kill_actor_processes(process)

    def _run_online_training_iteration(self, env):
        def normalize(values):
            return (values - np.mean(values)) / np.std(values)

        update_interval = self._params.actor_timesteps * self._params.actor_num
        if self.iteration_num % update_interval != 0:
            return

        s, a, r, non_terminal, s_next, log_prob, v_targets, advantages = \
            self._collect_experiences(self._actors)

        if self._state_preprocessor is not None:
            self._state_preprocessor.update(s)

        advantages = normalize(advantages)
        data = list(zip(s, a, r, non_terminal, s_next,
                        log_prob, v_targets, advantages))
        replay_buffer = ReplayBuffer()
        replay_buffer.append_all(data)

        buffer_iterator = BufferIterator(
            replay_buffer, batch_size=self._params.batch_size)
        for _ in range(self._params.epochs):
            for experiences, *_ in buffer_iterator:
                self._ppo_training(experiences)
            buffer_iterator.reset()

    def _launch_actor_processes(self, env):
        actors = self._build_ppo_actors(env,
                                        v_function=self._v_function,
                                        policy=self._policy,
                                        state_preprocessor=self._state_preprocessor)
        processes = []
        for actor in actors:
            p = mp.Process(target=actor, daemon=True)
            p.start()
            processes.append(p)
        return actors, processes

    def _kill_actor_processes(self, process):
        process.terminate()
        process.join()

    def _run_offline_training_iteration(self, buffer):
        raise NotImplementedError

    def _collect_experiences(self, actors):
        for actor in self._actors:
            if self._state_preprocessor is not None:
                actor.update_preprocessor_params(
                    self._state_preprocessor.get_parameters())
            actor.update_v_params(self._v_function.get_parameters())
            actor.update_policy_params(self._policy.get_parameters())

            actor.run_data_collection()

        results = []
        for actor in actors:
            result = actor.wait_data_collection()
            results.append(result)
        return (np.concatenate(item, axis=0) for item in unzip(results))

    def _ppo_training(self, experiences):
        (s, a, _, _, _, log_prob, v_target, advantage) = \
            marshall_experiences(experiences)

        alpha = (1.0 - self.iteration_num / self._params.total_timesteps) \
            if self._params.decrease_alpha else 1.0
        alpha = np.maximum(alpha, 0.0)

        # Optimize critic
        self._variables.s_current.d = s
        self._variables.a_current.d = a
        self._variables.log_prob.d = log_prob
        self._variables.v_target.d = v_target
        self._variables.advantage.d = advantage
        self._variables.alpha.d = alpha
        self._v_solver.set_learning_rate(
            self._params.learning_rate * alpha)
        self._policy_solver.set_learning_rate(
            self._params.learning_rate * alpha)

        # Optimize actor and value function
        self._v_solver.zero_grad()
        self._policy_solver.zero_grad()
        self._loss.forward()
        self._loss.backward()
        self._v_solver.update()
        self._policy_solver.update()

    def _compute_action(self, s):
        self._eval_state_var.d = np.expand_dims(s, axis=0)
        self._eval_action.forward(clear_buffer=True)
        action = np.squeeze(self._eval_action.d, axis=0)
        if self._env_info.is_discrete_action_env():
            return np.int(action)
        else:
            return action

    def _models(self):
        models = {}
        models[self._v_function.scope_name] = self._v_function
        models[self._policy.scope_name] = self._policy
        if self._state_preprocessor is not None:
            models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers['v_solver'] = self._v_solver
        solvers['policy_solver'] = self._policy_solver
        return solvers

    def _build_ppo_actors(self, env, v_function, policy, state_preprocessor):
        actors = []
        for i in range(self._params.actor_num):
            actor = _PPOActor(env=env,
                              env_info=self._env_info,
                              state_preprocessor=state_preprocessor,
                              v_function=v_function,
                              policy=policy,
                              params=self._params)
            actors.append(actor)
        return actors


class _PPOActor(object):
    def __init__(self, env, env_info, state_preprocessor, v_function, policy, params):
        # These variables will be copied when process is created
        self._env = env
        self._env_info = env_info
        self._state_preprocessor = state_preprocessor
        self._v_function = v_function
        self._policy = policy
        self._timesteps = params.actor_timesteps
        self._gamma = params.gamma
        self._lambda = params.lmb
        self._params = params

        # IPC communication variables
        self._disposed = mp.Value('i', False)
        self._task_start_event = mp.Event()
        self._task_finish_event = mp.Event()

        if state_preprocessor is not None:
            self._state_preprocessor_mp_arrays = new_mp_arrays_from_params(
                state_preprocessor.get_parameters())
        self._v_mp_arrays = new_mp_arrays_from_params(
            v_function.get_parameters())
        self._policy_mp_arrays = new_mp_arrays_from_params(
            policy.get_parameters())

        self._state = None

        obs_space = self._env.observation_space
        action_space = self._env.action_space

        MultiProcessingArrays = namedtuple('MultiProcessingArrays',
                                           ['state', 'action', 'reward', 'non_terminal', 'next_state', 'log_prob', 'v_target', 'advantage'])

        state_mp_array_shape = (self._timesteps, *obs_space.shape)
        state_mp_array = mp_array_from_np_array(
            np.empty(shape=state_mp_array_shape, dtype=obs_space.dtype))
        if env_info.is_discrete_action_env():
            action_mp_array_shape = (self._timesteps, 1)
            action_mp_array = mp_array_from_np_array(
                np.empty(shape=action_mp_array_shape, dtype=action_space.dtype))
        else:
            action_mp_array_shape = (self._timesteps, action_space.shape[0])
            action_mp_array = mp_array_from_np_array(
                np.empty(shape=action_mp_array_shape, dtype=action_space.dtype))

        scalar_mp_array_shape = (self._timesteps, 1)
        reward_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))
        non_terminal_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))
        next_state_mp_array = mp_array_from_np_array(
            np.empty(shape=state_mp_array_shape, dtype=obs_space.dtype))
        log_prob_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))
        v_target_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))
        advantage_mp_array = mp_array_from_np_array(
            np.empty(shape=scalar_mp_array_shape, dtype=np.float32))

        self._mp_arrays = MultiProcessingArrays(
            (state_mp_array, state_mp_array_shape, obs_space.dtype),
            (action_mp_array, action_mp_array_shape, action_space.dtype),
            (reward_mp_array, scalar_mp_array_shape, np.float32),
            (non_terminal_mp_array, scalar_mp_array_shape, np.float32),
            (next_state_mp_array, state_mp_array_shape, obs_space.dtype),
            (log_prob_mp_array, scalar_mp_array_shape, np.float32),
            (v_target_mp_array, scalar_mp_array_shape, np.float32),
            (advantage_mp_array, scalar_mp_array_shape, np.float32)
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

    def update_preprocessor_params(self, params):
        self._update_params(
            src=params, dest=self._state_preprocessor_mp_arrays)

    def update_v_params(self, params):
        self._update_params(src=params, dest=self._v_mp_arrays)

    def update_policy_params(self, params):
        self._update_params(src=params, dest=self._policy_mp_arrays)

    def _run_actor_loop(self):
        context._set_nnabla_context()
        self._env.seed(os.getpid())
        while (True):
            self._task_start_event.wait()
            if self._disposed.get_obj():
                break

            if self._state_preprocessor is not None:
                self._synchronize_preprocessor_params(
                    self._state_preprocessor.get_parameters())
            self._synchronize_v_params(self._v_function.get_parameters())
            self._synchronize_policy_params(self._policy.get_parameters())

            experiences, v_targets, advantages = self._run_data_collection()
            self._fill_result(experiences, v_targets, advantages)

            self._task_start_event.clear()
            self._task_finish_event.set()

    def _run_data_collection(self):
        experiences = []
        if self._state is None:
            state = self._env.reset()
        else:
            state = self._state
        state_var = nn.Variable((1, *state.shape))
        distribution = self._policy.pi(state_var)
        action_var, log_prob_var = distribution.sample_and_compute_log_prob()
        is_discrete_action = isinstance(
            self._env.action_space, gym.spaces.Discrete)
        for _ in range(self._timesteps):
            state_var.d = state
            nn.forward_all((action_var, log_prob_var))
            action = np.squeeze(action_var.d, axis=0).copy()
            if is_discrete_action:
                action = np.int(action)
            log_prob = np.squeeze(log_prob_var.d).copy()
            s_next, reward, done, info = self._env.step(action)
            # We don't treat done at max_episode_length as False.
            # In Swimmer, done must be treated as done even if max_episode_length is reached
            truncated = info.get('TimeLimit.truncated', False) and self._params.only_reset_if_truncated
            if done and not truncated:
                non_terminal = 0.0
            else:
                non_terminal = 1.0
            non_terminal = 0.0 if done else 1.0
            experience = (state, [action], reward,
                          non_terminal, s_next, [log_prob])
            experiences.append(experience)

            if done:
                state = self._env.reset()
            else:
                state = s_next
        self._state = state
        v_targets, advantages = self._compute_v_target_and_advantage(
            self._v_function, experiences)
        return experiences, v_targets, advantages

    def _fill_result(self, experiences, v_targets, advantages):
        def array_and_dtype(mp_arrays_item):
            return mp_arrays_item[0], mp_arrays_item[2]
        (s, a, r, non_terminal, s_next, log_prob) = marshall_experiences(experiences)
        np_to_mp_array(s, *array_and_dtype(self._mp_arrays.state))
        np_to_mp_array(a, *array_and_dtype(self._mp_arrays.action))
        np_to_mp_array(r, *array_and_dtype(self._mp_arrays.reward))
        np_to_mp_array(
            non_terminal, *array_and_dtype(self._mp_arrays.non_terminal))
        np_to_mp_array(s_next, *array_and_dtype(self._mp_arrays.next_state))
        np_to_mp_array(log_prob, *array_and_dtype(self._mp_arrays.log_prob))
        np_to_mp_array(v_targets, *array_and_dtype(self._mp_arrays.v_target))
        np_to_mp_array(advantages, *array_and_dtype(self._mp_arrays.advantage))

    def _update_params(self, src, dest):
        copy_params_to_mp_arrays(src, dest)

    def _compute_v_target_and_advantage(self, v_function, experiences):
        T = len(experiences)
        v_targets = []
        advantages = []
        advantage = 0

        v_current = None
        v_next = None
        state_shape = self._env.observation_space.shape
        state_var = nn.Variable((1, *state_shape))
        v_var = v_function.v(state_var)
        for t in reversed(range(T)):
            s_current, _, r, non_terminal, s_next, _ = experiences[t]

            state_var.d = s_current
            v_var.forward()
            v_current = np.squeeze(v_var.d).copy()

            if v_next is None:
                state_var.d = s_next
                v_var.forward()
                v_next = np.squeeze(v_var.d).copy()

            delta = r + self._gamma * non_terminal * v_next - v_current
            advantage = np.float32(
                delta + self._gamma * self._lambda * non_terminal * advantage)
            # A = Q - V, V = E[Q] -> v_target = A + V
            v_target = advantage + v_current

            v_targets.insert(0, v_target)
            advantages.insert(0, advantage)

            v_next = v_current
        return np.asarray(v_targets, dtype=np.float32), np.asarray(advantages, dtype=np.float32)

    def _synchronize_preprocessor_params(self, params):
        self._synchronize_params(src=self._state_preprocessor_mp_arrays,
                                 dest=params)

    def _synchronize_v_params(self, params):
        self._synchronize_params(src=self._v_mp_arrays, dest=params)

    def _synchronize_policy_params(self, params):
        self._synchronize_params(src=self._policy_mp_arrays, dest=params)

    def _synchronize_params(self, src, dest):
        copy_mp_arrays_to_params(src, dest)
