import nnabla as nn
import nnabla.solvers as NS

from dataclasses import dataclass
from collections import namedtuple

import numpy as np

import multiprocessing as mp

import os

import gym
from typing import Optional, Union

import nnabla_rl.preprocessors as RP
import nnabla_rl.model_trainers as MT
import nnabla_rl.environment_explorers as EE
import nnabla_rl.utils.context as context
from nnabla_rl.builders import ModelBuilder, SolverBuilder, PreprocessorBuilder
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.model_trainers.model_trainer import TrainingBatch
from nnabla_rl.models import \
    PPOSharedFunctionHead, PPOAtariPolicy, PPOAtariVFunction, \
    PPOMujocoPolicy, PPOMujocoVFunction, \
    StochasticPolicy, VFunction, \
    Model
from nnabla_rl.algorithms.common_utils import \
    compute_v_target_and_advantage, _StatePreprocessedPolicy, _StatePreprocessedVFunction
from nnabla_rl.algorithm import Algorithm, AlgorithmParam, eval_api
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers import BufferIterator
from nnabla_rl.utils.data import marshall_experiences, unzip
from nnabla_rl.utils.multiprocess import (mp_to_np_array, np_to_mp_array,
                                          mp_array_from_np_array, new_mp_arrays_from_params,
                                          copy_mp_arrays_to_params, copy_params_to_mp_arrays)
from nnabla_rl.utils.reproductions import set_global_seed


class DefaultPolicyBuilder(ModelBuilder):
    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> Model:
        if env_info.is_discrete_action_env():
            # scope name is same as that of v-function -> parameter is shared across models automatically
            return self._build_shared_policy("shared", env_info, algorithm_params)
        else:
            return self._build_mujoco_policy(scope_name, env_info, algorithm_params)

    def _build_shared_policy(self, scope_name, env_info, algorithm_params, **kwargs):
        _shared_function_head = PPOSharedFunctionHead(scope_name=scope_name,
                                                      state_shape=env_info.state_shape,
                                                      action_dim=env_info.action_dim)
        return PPOAtariPolicy(scope_name=scope_name, action_dim=env_info.action_dim, head=_shared_function_head)

    def _build_mujoco_policy(self, scope_name, env_info, algorithm_params, **kwargs):
        policy = PPOMujocoPolicy(scope_name=scope_name, action_dim=env_info.action_dim)
        return policy


class DefaultVFunctionBuilder(ModelBuilder):
    def build_model(self,
                    scope_name: str,
                    env_info: EnvironmentInfo,
                    algorithm_params: AlgorithmParam,
                    **kwargs) -> Model:
        if env_info.is_discrete_action_env():
            # scope name is same as that of policy -> parameter is shared across models automatically
            return self._build_shared_v_function("shared", env_info, algorithm_params)
        else:
            return self._build_mujoco_v_function(scope_name, env_info, algorithm_params)

    def _build_shared_v_function(self, scope_name, env_info, algorithm_params, **kwargs):
        _shared_function_head = PPOSharedFunctionHead(scope_name=scope_name,
                                                      state_shape=env_info.state_shape,
                                                      action_dim=env_info.action_dim)
        return PPOAtariVFunction(scope_name=scope_name, head=_shared_function_head)

    def _build_mujoco_v_function(self, scope_name, env_info, algorithm_params, **kwargs):
        v_function = PPOMujocoVFunction(scope_name=scope_name)
        return v_function


class DefaultSolverBuilder(SolverBuilder):
    def build_solver(self,
                     env_info: EnvironmentInfo,
                     algorithm_params: AlgorithmParam,
                     **kwargs) -> nn.solver.Solver:
        return NS.Adam(alpha=algorithm_params.learning_rate, eps=1e-5)


class DefaultPreprocessorBuilder(PreprocessorBuilder):
    def build_preprocessor(self,
                           scope_name: str,
                           env_info: EnvironmentInfo,
                           algorithm_params: AlgorithmParam,
                           **kwargs):
        return RP.RunningMeanNormalizer('preprocessor', env_info.state_shape, value_clip=(-5.0, 5.0))


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
    timelimit_as_terminal: bool = False
    seed: int = -1
    preprocess_state: bool = True

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

    def __init__(self, env_or_env_info: Union[gym.Env, EnvironmentInfo],
                 params: PPOParam = PPOParam(),
                 v_function_builder: ModelBuilder = DefaultVFunctionBuilder(),
                 v_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 policy_builder: ModelBuilder = DefaultPolicyBuilder(),
                 policy_solver_builder: SolverBuilder = DefaultSolverBuilder(),
                 state_preprocessor_builder: Optional[PreprocessorBuilder] = DefaultPreprocessorBuilder()):
        self._gpu_id = context._gpu_id
        # Prevent setting context by the Algorithm class
        if 0 <= self._gpu_id:
            context._gpu_id = -1
        super(PPO, self).__init__(env_or_env_info, params=params)

        v_function = v_function_builder('v', self._env_info, self._params)
        policy = policy_builder('pi', self._env_info, self._params)
        if self._params.preprocess_state:
            preprocessor = state_preprocessor_builder('preprocessor', self._env_info, self._params)
            self._v_function = _StatePreprocessedVFunction(v_function=v_function, preprocessor=preprocessor)
            self._policy = _StatePreprocessedPolicy(policy=policy, preprocessor=preprocessor)
            self._state_preprocessor = preprocessor
        else:
            self._v_function = v_function
            self._policy = policy
            self._state_preprocessor = None

        assert isinstance(self._policy, StochasticPolicy)
        assert isinstance(self._v_function, VFunction)

        self._policy_solver = {self._policy.scope_name: policy_solver_builder(self._env_info, self._params)}
        self._v_function_solver = {self._v_function.scope_name: v_solver_builder(self._env_info, self._params)}

        self._actors = None
        self._actor_processes = []

    @eval_api
    def compute_eval_action(self, state):
        if context._gpu_id < 0 and 0 <= self._gpu_id:
            context._gpu_id = self._gpu_id
            context._set_nnabla_context()
        return self._compute_action(state)

    def _before_training_start(self, env_or_buffer):
        if not self._is_env(env_or_buffer):
            raise ValueError('PPO only supports online training')
        context._gpu_id = self._gpu_id
        env = env_or_buffer

        # FIXME: This setup is a workaround for creating underliying model parameters
        # If the parameter is not created, the multiprocessable array (created in launch_actor_processes)
        # will be empty and the agent does not learn anything
        self._setup_policy_training(env_or_buffer)
        self._setup_v_function_training(env_or_buffer)

        self._actors, self._actor_processes = self._launch_actor_processes(env)
        context._set_nnabla_context()

        # Setup again here to use gpu (if it is set)
        self._policy_trainer = self._setup_policy_training(env_or_buffer)
        self._v_function_trainer = self._setup_v_function_training(env_or_buffer)

    def _setup_policy_training(self, env_or_buffer):
        policy_trainer_params = MT.policy_trainers.PPOPolicyTrainerParam(
            entropy_coefficient=self._params.entropy_coefficient
        )
        policy_trainer = MT.policy_trainers.PPOPolicyTrainer(
            env_info=self._env_info,
            params=policy_trainer_params)

        training = MT.model_trainer.Training()
        policy_trainer.setup_training(self._policy, self._policy_solver, training)
        return policy_trainer

    def _setup_v_function_training(self, env_or_buffer):
        # training input/loss variables
        v_function_trainer_params = MT.v_value_trainers.SquaredTDVFunctionTrainerParam(
            reduction_method='mean',
            v_loss_scalar=self._params.value_coefficient
        )

        v_function_trainer = MT.v_value_trainers.SquaredTDVFunctionTrainer(
            env_info=self._env_info,
            params=v_function_trainer_params)

        training = MT.v_value_trainings.MonteCarloVValueTraining()
        v_function_trainer.setup_training(self._v_function, self._v_function_solver, training)
        return v_function_trainer

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

        if self._params.preprocess_state:
            self._state_preprocessor.update(s)

        advantages = normalize(advantages)
        data = list(zip(s, a, r, non_terminal, s_next, log_prob, v_targets, advantages))
        replay_buffer = ReplayBuffer()
        replay_buffer.append_all(data)

        buffer_iterator = BufferIterator(replay_buffer, batch_size=self._params.batch_size)
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
            actor.update_v_params(self._v_function.get_parameters())
            actor.update_policy_params(self._policy.get_parameters())
            if self._params.preprocess_state:
                actor.update_state_preprocessor_params(self._state_preprocessor.get_parameters())

            actor.run_data_collection()

        results = []
        for actor in actors:
            result = actor.wait_data_collection()
            results.append(result)
        return (np.concatenate(item, axis=0) for item in unzip(results))

    def _ppo_training(self, experiences):
        if self._params.decrease_alpha:
            alpha = (1.0 - self.iteration_num / self._params.total_timesteps)
            alpha = np.maximum(alpha, 0.0)
        else:
            alpha = 1.0

        (s, a, _, _, _, log_prob, v_target, advantage) = marshall_experiences(experiences)

        extra = {}
        extra['log_prob'] = log_prob
        extra['advantage'] = advantage
        extra['alpha'] = alpha
        extra['v_target'] = v_target
        batch = TrainingBatch(batch_size=len(experiences),
                              s_current=s,
                              a_current=a,
                              extra=extra)

        self._policy_trainer.set_learning_rate(self._params.learning_rate * alpha)
        self._policy_trainer.train(batch)
        self._v_function_trainer.set_learning_rate(self._params.learning_rate * alpha)
        self._v_function_trainer.train(batch)

    def _compute_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._policy.pi(self._eval_state_var)
            self._eval_action = distribution.sample()
        self._eval_state_var.d = s
        self._eval_action.forward()
        action = np.squeeze(self._eval_action.d, axis=0)
        if self._env_info.is_discrete_action_env():
            return np.int(action)
        else:
            return action

    def _models(self):
        models = {}
        models[self._v_function.scope_name] = self._v_function
        models[self._policy.scope_name] = self._policy
        if self._params.preprocess_state and isinstance(self._state_preprocessor, Model):
            models[self._state_preprocessor.scope_name] = self._state_preprocessor
        return models

    def _solvers(self):
        solvers = {}
        solvers.update(self._policy_solver)
        solvers.update(self._v_function_solver)
        return solvers

    def _build_ppo_actors(self, env, v_function, policy, state_preprocessor):
        actors = []
        for i in range(self._params.actor_num):
            actor = _PPOActor(
                actor_num=i,
                env=env,
                env_info=self._env_info,
                v_function=v_function,
                policy=policy,
                state_preprocessor=state_preprocessor,
                params=self._params)
            actors.append(actor)
        return actors


class _PPOActor(object):
    def __init__(self, actor_num, env, env_info, v_function, policy, state_preprocessor, params):
        # These variables will be copied when process is created
        self._actor_num = actor_num
        self._env = env
        self._env_info = env_info
        self._v_function = v_function
        self._policy = policy
        self._state_preprocessor = state_preprocessor
        self._timesteps = params.actor_timesteps
        self._gamma = params.gamma
        self._lambda = params.lmb
        self._params = params

        # IPC communication variables
        self._disposed = mp.Value('i', False)
        self._task_start_event = mp.Event()
        self._task_finish_event = mp.Event()

        self._v_mp_arrays = new_mp_arrays_from_params(v_function.get_parameters())
        self._policy_mp_arrays = new_mp_arrays_from_params(policy.get_parameters())
        if self._params.preprocess_state:
            self._state_preprocessor_mp_arrays = new_mp_arrays_from_params(state_preprocessor.get_parameters())

        explorer_params = EE.RawPolicyExplorerParam(
            initial_step_num=0,
            timelimit_as_terminal=self._params.timelimit_as_terminal
        )
        self._environment_explorer = EE.RawPolicyExplorer(policy_action_selector=self._compute_action,
                                                          env_info=self._env_info,
                                                          params=explorer_params)

        obs_space = self._env.observation_space
        action_space = self._env.action_space

        MultiProcessingArrays = namedtuple('MultiProcessingArrays',
                                           ['state', 'action', 'reward', 'non_terminal',
                                            'next_state', 'log_prob', 'v_target', 'advantage'])

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

    def update_v_params(self, params):
        self._update_params(src=params, dest=self._v_mp_arrays)

    def update_policy_params(self, params):
        self._update_params(src=params, dest=self._policy_mp_arrays)

    def update_state_preprocessor_params(self, params):
        self._update_params(src=params, dest=self._state_preprocessor_mp_arrays)

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

            self._synchronize_v_params(self._v_function.get_parameters())
            self._synchronize_policy_params(self._policy.get_parameters())
            if self._params.preprocess_state:
                self._synchronize_preprocessor_params(self._state_preprocessor.get_parameters())

            experiences, v_targets, advantages = self._run_data_collection()
            self._fill_result(experiences, v_targets, advantages)

            self._task_start_event.clear()
            self._task_finish_event.set()

    def _run_data_collection(self):
        experiences = self._environment_explorer.step(self._env, n=self._timesteps)
        experiences = [(s, a, r, non_terminal, s_next, info['log_prob'])
                       for (s, a, r, non_terminal, s_next, info) in experiences]
        v_targets, advantages = compute_v_target_and_advantage(
            self._v_function, experiences, gamma=self._gamma, lmb=self._lambda)
        return experiences, v_targets, advantages

    def _compute_action(self, s):
        s = np.expand_dims(s, axis=0)
        if not hasattr(self, '_eval_state_var'):
            self._eval_state_var = nn.Variable(s.shape)
            distribution = self._policy.pi(self._eval_state_var)
            self._eval_action, self._eval_log_prob = distribution.sample_and_compute_log_prob()
        self._eval_state_var.d = s
        nn.forward_all([self._eval_action, self._eval_log_prob])
        action = np.squeeze(self._eval_action.d, axis=0)
        log_prob = np.squeeze(self._eval_log_prob.d, axis=0)
        info = {}
        info['log_prob'] = log_prob
        if self._env_info.is_discrete_action_env():
            return np.int(action), info
        else:
            return action, info

    def _fill_result(self, experiences, v_targets, advantages):
        def array_and_dtype(mp_arrays_item):
            return mp_arrays_item[0], mp_arrays_item[2]
        (s, a, r, non_terminal, s_next, log_prob) = marshall_experiences(experiences)
        np_to_mp_array(s, *array_and_dtype(self._mp_arrays.state))
        np_to_mp_array(a, *array_and_dtype(self._mp_arrays.action))
        np_to_mp_array(r, *array_and_dtype(self._mp_arrays.reward))
        np_to_mp_array(non_terminal, *array_and_dtype(self._mp_arrays.non_terminal))
        np_to_mp_array(s_next, *array_and_dtype(self._mp_arrays.next_state))
        np_to_mp_array(log_prob, *array_and_dtype(self._mp_arrays.log_prob))
        np_to_mp_array(v_targets, *array_and_dtype(self._mp_arrays.v_target))
        np_to_mp_array(advantages, *array_and_dtype(self._mp_arrays.advantage))

    def _update_params(self, src, dest):
        copy_params_to_mp_arrays(src, dest)

    def _synchronize_v_params(self, params):
        self._synchronize_params(src=self._v_mp_arrays, dest=params)

    def _synchronize_policy_params(self, params):
        self._synchronize_params(src=self._policy_mp_arrays, dest=params)

    def _synchronize_preprocessor_params(self, params):
        self._synchronize_params(src=self._state_preprocessor_mp_arrays, dest=params)

    def _synchronize_params(self, src, dest):
        copy_mp_arrays_to_params(src, dest)
