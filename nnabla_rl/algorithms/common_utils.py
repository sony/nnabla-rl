# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from abc import ABCMeta, abstractmethod
from typing import Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla_rl.functions as RF
from nnabla_rl.algorithm import eval_api
from nnabla_rl.distributions.distribution import Distribution
from nnabla_rl.environments.environment_info import EnvironmentInfo
from nnabla_rl.models import (DeterministicDecisionTransformer, DeterministicDynamics, DeterministicPolicy,
                              FactoredContinuousQFunction, Model, QFunction, RewardFunction,
                              StochasticDecisionTransformer, StochasticPolicy, VFunction)
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.typing import Experience, State
from nnabla_rl.utils.data import add_batch_dimension, marshal_experiences, set_data_to_variable
from nnabla_rl.utils.misc import create_variable, create_variables

DecisionTransformerModel = Union[StochasticDecisionTransformer, DeterministicDecisionTransformer]


def _get_shape(state: State) -> Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    if isinstance(state, tuple):
        shape = tuple(np.atleast_1d(s).shape for s in state)
    else:
        shape = np.atleast_1d(state).shape
    return shape


def has_batch_dimension(state: State, env_info: EnvironmentInfo):
    fed_state_shape = _get_shape(state)
    env_state_shape = env_info.state_shape

    return not (fed_state_shape == env_state_shape)


def compute_v_target_and_advantage(v_function: VFunction,
                                   experiences: Sequence[Experience],
                                   gamma: float = 0.99,
                                   lmb: float = 0.97) -> Tuple[np.ndarray, np.ndarray]:
    """Compute value target and advantage by using Generalized Advantage
    Estimation (GAE)

    Args:
        v_function (M.VFunction): value function
        experiences (Sequence[Experience]): list of experience.
            experience should have [state_current, action, reward, non_terminal, state_next]
        gamma (float): discount rate
        lmb (float): lambda
        preprocess_func (callable): preprocess function of states
    Returns:
        Tuple[np.ndarray, np.ndarray]: target of value and advantage
    Ref:
        https://arxiv.org/pdf/1506.02438.pdf
    """
    assert isinstance(v_function, VFunction), "Invalid v_function"

    T = len(experiences)
    v_targets: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantages: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantage: np.float32 = np.float32(0.)

    v_current = None
    v_next = None
    s_var = create_variable(1, _get_shape(experiences[0][0]))
    v = v_function.v(s_var)  # build graph

    for t in reversed(range(T)):
        s_current, _, r, non_terminal, s_next, *_ = experiences[t]

        # predict current v
        set_data_to_variable(s_var, s_current)
        v.forward()
        v_current = np.squeeze(v.d)

        if v_next is None:
            set_data_to_variable(s_var, s_next)
            v.forward()
            v_next = np.squeeze(v.d)

        delta = r + gamma * non_terminal * v_next - v_current
        advantage = np.float32(delta + gamma * lmb * non_terminal * advantage)
        # A = Q - V, V = E[Q] -> v_target = A + V
        v_target = advantage + v_current

        v_targets[t] = v_target
        advantages[t] = advantage

        v_next = v_current

    return np.array(v_targets, dtype=np.float32), np.array(advantages, dtype=np.float32)


def compute_average_v_target_and_advantage(v_function: VFunction,
                                           experiences: Sequence[Experience],
                                           lmb=0.95):
    ''' Compute value target and advantage by using Average Reward Criterion
    See: https://arxiv.org/pdf/2106.07329.pdf

    Args:
        v_function (VFunction): value function
        experiences (Sequence[Experience]): list of experience.
            experience should have [state_current, action, reward, non_terminal, state_next]
        lmb (float): lambda
    Returns:
        Tuple[np.ndarray, np.ndarray]: target of value and advantage
    '''
    assert isinstance(v_function, VFunction), "Invalid v_function"
    T = len(experiences)
    v_targets: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantages: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantage: np.float32 = np.float32(0.)

    v_current = None
    v_next = None
    s_var = create_variable(1, _get_shape(experiences[0][0]))
    v = v_function.v(s_var)  # build graph

    _, _, batch_r, *_ = marshal_experiences(experiences)
    average_r = np.mean(batch_r)

    for t in reversed(range(T)):
        s_current, _, r, non_terminal, s_next, *_ = experiences[t]

        # predict current v
        set_data_to_variable(s_var, s_current)
        v.forward()
        v_current = np.squeeze(v.d)

        if v_next is None:
            set_data_to_variable(s_var, s_next)
            v.forward()
            v_next = np.squeeze(v.d)

        delta = (r - average_r) + non_terminal * v_next - v_current
        advantage = np.float32(delta + lmb * non_terminal * advantage)
        # A = Q - V, V = E[Q] -> v_target = A + V
        v_target = advantage + v_current

        v_targets[t] = v_target
        advantages[t] = advantage

        v_next = v_current

    return np.array(v_targets, dtype=np.float32), np.array(advantages, dtype=np.float32)


class _StatePreprocessedVFunction(VFunction):
    _v_function: VFunction
    _preprocessor: Preprocessor

    def __init__(self, v_function: VFunction, preprocessor: Preprocessor):
        super(_StatePreprocessedVFunction, self).__init__(v_function.scope_name)
        self._v_function = v_function
        self._preprocessor = preprocessor

    def v(self, s: nn.Variable):
        preprocessed_state = self._preprocessor.process(s)
        return self._v_function.v(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> '_StatePreprocessedVFunction':
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedVFunction)
        copied._v_function._scope_name = new_scope_name
        return copied

    def is_recurrent(self) -> bool:
        return self._v_function.is_recurrent()

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._v_function.internal_state_shapes()

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        return self._v_function.set_internal_states(states)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        return self._v_function.get_internal_states()


class _StatePreprocessedDeterministicPolicy(DeterministicPolicy):
    _policy: DeterministicPolicy
    _preprocessor: Preprocessor

    def __init__(self, policy: DeterministicPolicy, preprocessor: Preprocessor):
        super(_StatePreprocessedDeterministicPolicy, self).__init__(policy.scope_name)
        self._policy = policy
        self._preprocessor = preprocessor

    def pi(self, s: nn.Variable) -> nn.Variable:
        preprocessed_state = self._preprocessor.process(s)
        return self._policy.pi(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> '_StatePreprocessedDeterministicPolicy':
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedDeterministicPolicy)
        copied._policy._scope_name = new_scope_name
        return copied

    def is_recurrent(self) -> bool:
        return self._policy.is_recurrent()

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._policy.internal_state_shapes()

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        return self._policy.set_internal_states(states)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        return self._policy.get_internal_states()


class _StatePreprocessedStochasticPolicy(StochasticPolicy):
    _policy: StochasticPolicy
    _preprocessor: Preprocessor

    def __init__(self, policy: StochasticPolicy, preprocessor: Preprocessor):
        super(_StatePreprocessedStochasticPolicy, self).__init__(policy.scope_name)
        self._policy = policy
        self._preprocessor = preprocessor

    def pi(self, s: nn.Variable) -> Distribution:
        preprocessed_state = self._preprocessor.process(s)
        return self._policy.pi(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> '_StatePreprocessedStochasticPolicy':
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedStochasticPolicy)
        copied._policy._scope_name = new_scope_name
        return copied

    def is_recurrent(self) -> bool:
        return self._policy.is_recurrent()

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._policy.internal_state_shapes()

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        return self._policy.set_internal_states(states)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        return self._policy.get_internal_states()


class _StatePreprocessedRewardFunction(RewardFunction):
    _reward_function: RewardFunction
    _preprocessor: Preprocessor

    def __init__(self, reward_function: RewardFunction, preprocessor: Preprocessor):
        super(_StatePreprocessedRewardFunction, self).__init__(reward_function.scope_name)
        self._reward_function = reward_function
        self._preprocessor = preprocessor

    def r(self, s_current: nn.Variable, a_current: nn.Variable, s_next: nn.Variable) -> nn.Variable:
        preprocessed_state_current = self._preprocessor.process(s_current)
        preprocessed_state_next = self._preprocessor.process(s_next)
        return self._reward_function.r(preprocessed_state_current, a_current, preprocessed_state_next)

    def deepcopy(self, new_scope_name: str) -> '_StatePreprocessedRewardFunction':
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedRewardFunction)
        copied._reward_function._scope_name = new_scope_name
        return copied

    def is_recurrent(self) -> bool:
        return self._reward_function.is_recurrent()

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._reward_function.internal_state_shapes()

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        return self._reward_function.set_internal_states(states)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        return self._reward_function.get_internal_states()


class _StatePreprocessedQFunction(QFunction):
    _q_function: QFunction
    _preprocessor: Preprocessor

    def __init__(self, q_function: QFunction, preprocessor: Preprocessor):
        super(_StatePreprocessedQFunction, self).__init__(q_function.scope_name)
        self._q_function = q_function
        self._preprocessor = preprocessor

    def q(self, s: nn.Variable, a: nn.Variable):
        preprocessed_state = self._preprocessor.process(s)
        return self._q_function.q(preprocessed_state, a)

    def all_q(self, s: nn.Variable) -> nn.Variable:
        preprocessed_state = self._preprocessor.process(s)
        return self._q_function.all_q(preprocessed_state)

    def max_q(self, s: nn.Variable) -> nn.Variable:
        preprocessed_state = self._preprocessor.process(s)
        return self._q_function.max_q(preprocessed_state)

    def argmax_q(self, s: nn.Variable) -> nn.Variable:
        preprocessed_state = self._preprocessor.process(s)
        return self._q_function.argmax_q(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> '_StatePreprocessedQFunction':
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedQFunction)
        copied._q_function._scope_name = new_scope_name
        return copied

    def is_recurrent(self) -> bool:
        return self._q_function.is_recurrent()

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._q_function.internal_state_shapes()

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        return self._q_function.set_internal_states(states)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        return self._q_function.get_internal_states()


M = TypeVar('M', bound=Model)


class _ActionSelector(Generic[M], metaclass=ABCMeta):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _model: M

    def __init__(self, env_info: EnvironmentInfo, model: M):
        self._env_info = env_info
        self._model = model
        self._batch_size = 1

    @eval_api
    def __call__(self, s: Union[np.ndarray, Tuple[np.ndarray, ...]], *, begin_of_episode: bool = False, extra_info={}):
        if not has_batch_dimension(s, self._env_info):
            s = add_batch_dimension(s)
        batch_size = len(s[0]) if self._env_info.is_tuple_state_env() else len(s)
        if not hasattr(self, '_eval_state_var') or batch_size != self._batch_size:
            # Variable creation
            self._eval_state_var = create_variable(batch_size, self._env_info.state_shape)
            if self._model.is_recurrent():
                self._rnn_internal_states = create_variables(batch_size, self._model.internal_state_shapes())
                self._model.set_internal_states(self._rnn_internal_states)
            self._action = self._compute_action(self._eval_state_var)
            if self._model.is_recurrent():
                self._model.reset_internal_states()
            self._batch_size = batch_size
        # Forward network
        if self._model.is_recurrent() and begin_of_episode:
            self._model.reset_internal_states()
        set_data_to_variable(self._eval_state_var, s)
        if self._model.is_recurrent():
            prev_rnn_states = self._model.get_internal_states()
            for key in self._rnn_internal_states.keys():
                # copy internal states of previous iteration
                self._rnn_internal_states[key].d = prev_rnn_states[key].d
        if self._env_info.is_tuple_action_env():
            nn.forward_all(self._action, clear_no_need_grad=True)
            action = tuple(np.squeeze(a.d, axis=0) if batch_size == 1 else a.d for a in self._action)
        else:
            self._action.forward(clear_no_need_grad=True)
            # No need to save internal states
            action = np.squeeze(self._action.d, axis=0) if batch_size == 1 else self._action.d
        return action, {}

    @abstractmethod
    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        raise NotImplementedError


class _DecisionTransformerActionSelector(_ActionSelector[DecisionTransformerModel]):
    def __init__(self, env_info: EnvironmentInfo, decision_transformer: DecisionTransformerModel,
                 max_timesteps: int, context_length: int, target_return: float, reward_scale: float):
        super().__init__(env_info, decision_transformer)
        self._max_timesteps = max_timesteps
        self._context_length = context_length
        self._target_return = target_return
        self._reward_scale = reward_scale

    def __call__(self, s: Union[np.ndarray, Tuple[np.ndarray, ...]], *, begin_of_episode: bool = False, extra_info={}):
        if self._env_info.is_tuple_state_env():
            raise NotImplementedError('Tuple env not supported')

        if not has_batch_dimension(s, self._env_info):
            s = add_batch_dimension(s)
        batch_size = len(s)

        if not hasattr(self, '_eval_states') or batch_size != self._batch_size:
            self._eval_states = np.empty(shape=(batch_size, self._context_length, *self._env_info.state_shape))
            self._eval_actions = np.empty(shape=(batch_size, self._context_length, *self._env_info.action_shape))
            self._eval_rtgs = np.empty(shape=(batch_size, self._context_length, 1))
            self._eval_timesteps = np.zeros(shape=(batch_size, 1, 1))

        if begin_of_episode:
            self._eval_timesteps[...] = 0

        t = int(self._eval_timesteps)
        T = min(t, self._context_length - 1)
        self._eval_states[:, T, ...] = s
        if t == 0:
            self._eval_rtgs[:, T, ...] = self._target_return * self._reward_scale
        else:
            reward = extra_info['reward'] * self._reward_scale
            self._eval_rtgs[:, T, ...] = self._eval_rtgs[:, T-1, ...] - reward

        with nn.auto_forward():
            states_var = nn.Variable.from_numpy_array(self._eval_states[:, 0:T+1, ...])
            if begin_of_episode:
                actions_var = None
            else:
                if self._context_length <= t:
                    actions_var = nn.Variable.from_numpy_array(self._eval_actions[:, 0:T+1, ...])
                else:
                    actions_var = nn.Variable.from_numpy_array(self._eval_actions[:, 0:T, ...])
            rtgs_var = nn.Variable.from_numpy_array(self._eval_rtgs[:, 0:T+1, ...])
            timesteps_var = nn.Variable.from_numpy_array(self._eval_timesteps)

            if isinstance(self._model, DeterministicDecisionTransformer):
                actions = self._model.pi(states_var, actions_var, rtgs_var, timesteps_var)
                action = np.squeeze(actions.d[:, T, :], axis=0)
            else:
                pi = self._model.pi(states_var, actions_var, rtgs_var, timesteps_var)
                actions = cast(nn.Variable, pi.sample())
                action = np.squeeze(actions.d[:, T, :], axis=0)

        if T == self._context_length - 1:
            self._eval_states = np.roll(self._eval_states, shift=-1, axis=1)
            self._eval_rtgs = np.roll(self._eval_rtgs, shift=-1, axis=1)
        if self._context_length <= t:
            self._eval_actions = np.roll(self._eval_actions, shift=-1, axis=1)

        self._eval_actions[:, T, ...] = action
        self._eval_timesteps[...] = min(cast(int, self._eval_timesteps + 1), self._max_timesteps)

        return action

    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        raise NotImplementedError


class _GreedyActionSelector(_ActionSelector[QFunction]):
    def __init__(self, env_info: EnvironmentInfo, q_function: QFunction):
        super().__init__(env_info, q_function)
        self._env_info = env_info
        self._q = q_function

    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        return self._q.argmax_q(self._eval_state_var)


class _StochasticPolicyActionSelector(_ActionSelector[StochasticPolicy]):
    def __init__(self, env_info, policy: StochasticPolicy, deterministic: bool = True):
        super().__init__(env_info, policy)
        self._deterministic = deterministic

    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        distribution = self._model.pi(self._eval_state_var)

        if self._deterministic:
            return distribution.choose_probable()
        else:
            return distribution.sample()


class _DeterministicPolicyActionSelector(_ActionSelector[DeterministicPolicy]):
    def __init__(self, env_info, policy: DeterministicPolicy):
        super().__init__(env_info, policy)

    def _compute_action(self, state_var: nn.Variable) -> nn.Variable:
        return self._model.pi(self._eval_state_var)


class _StatePredictor(Generic[M], metaclass=ABCMeta):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _model: M

    def __init__(self, env_info: EnvironmentInfo, model: M):
        self._env_info = env_info
        self._model = model
        self._batch_size = 1

    @eval_api
    def __call__(self,
                 s: Union[np.ndarray, Tuple[np.ndarray, ...]],
                 a: np.ndarray,
                 *,
                 begin_of_episode: bool = False):
        if not has_batch_dimension(s, self._env_info):
            s = add_batch_dimension(s)
        if not has_batch_dimension(a, self._env_info):
            a = cast(np.ndarray, add_batch_dimension(a))
        batch_size = len(s[0]) if self._env_info.is_tuple_state_env() else len(s)
        if not hasattr(self, '_eval_state_var') or batch_size != self._batch_size:
            # Variable creation
            self._eval_state_var = create_variable(batch_size, self._env_info.state_shape)
            self._eval_action_var = create_variable(batch_size, self._env_info.action_shape)
            if self._model.is_recurrent():
                self._rnn_internal_states = create_variables(batch_size, self._model.internal_state_shapes())
                self._model.set_internal_states(self._rnn_internal_states)
            self._next_state = self._compute_next_state(self._eval_state_var, self._eval_action_var)
            if self._model.is_recurrent():
                self._model.reset_internal_states()
            self._batch_size = batch_size
        # Forward network
        if self._model.is_recurrent() and begin_of_episode:
            self._model.reset_internal_states()
        set_data_to_variable(self._eval_state_var, s)
        set_data_to_variable(self._eval_action_var, a)
        if self._model.is_recurrent():
            prev_rnn_states = self._model.get_internal_states()
            for key in self._rnn_internal_states.keys():
                # copy internal states of previous iteration
                self._rnn_internal_states[key].d = prev_rnn_states[key].d
        self._next_state.forward(clear_no_need_grad=True)
        # No need to save internal states
        next_state = np.squeeze(self._next_state.d, axis=0) if batch_size == 1 else self._next_state.d
        return next_state, {}

    @abstractmethod
    def _compute_next_state(self, state_var: nn.Variable, action_var: nn.Variable) -> nn.Variable:
        raise NotImplementedError


class _DeterministicStatePredictor(_StatePredictor[DeterministicDynamics]):
    def __init__(self, env_info: EnvironmentInfo, dynamics: DeterministicDynamics):
        super().__init__(env_info, dynamics)
        self._dynamics = dynamics

    def _compute_next_state(self, state_var: nn.Variable, action_var: nn.Variable) -> nn.Variable:
        return self._dynamics.next_state(state_var, action_var)


class _InfluenceMetricsEvaluator:
    """Influence metrics evaluator.

    See details at https://arxiv.org/abs/2206.13901

    Args:
        env_info (EnvironmentInfo): Environment infomation.
        q_function (FactoredContinuousQFunction): Factored Q-function for continuous action.
    """
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _env_info: EnvironmentInfo
    _q_function: FactoredContinuousQFunction

    def __init__(self, env_info: EnvironmentInfo, q_function: FactoredContinuousQFunction):
        self._env_info = env_info
        self._q_function = q_function
        self._batch_size = 1

    @eval_api
    def __call__(self, s: Union[np.ndarray, Tuple[np.ndarray, ...]], a: np.ndarray, *, begin_of_episode: bool = False) \
            -> Tuple[np.ndarray, Dict]:
        if not has_batch_dimension(s, self._env_info):
            s = add_batch_dimension(s)
            a = cast(np.ndarray, add_batch_dimension(a))
        batch_size = len(s[0]) if self._env_info.is_tuple_state_env() else len(s)
        if not hasattr(self, '_eval_state_var') or batch_size != self._batch_size:
            # Variable creation
            self._batch_size = batch_size
            self._eval_state_var = create_variable(batch_size, self._env_info.state_shape)
            self._eval_action_var = create_variable(batch_size, self._env_info.action_shape)
            if self._q_function.is_recurrent():
                self._rnn_internal_states = create_variables(batch_size, self._q_function.internal_state_shapes())
                self._q_function.set_internal_states(self._rnn_internal_states)
            self._metrics = self._compute_influence_metrics(self._eval_state_var, self._eval_action_var)
            if self._q_function.is_recurrent():
                self._q_function.reset_internal_states()
        # Forward network
        if self._q_function.is_recurrent() and begin_of_episode:
            self._q_function.reset_internal_states()
        set_data_to_variable(self._eval_state_var, s)
        if self._q_function.is_recurrent():
            prev_rnn_states = self._q_function.get_internal_states()
            for key in self._rnn_internal_states.keys():
                # copy internal states of previous iteration
                self._rnn_internal_states[key].d = prev_rnn_states[key].d
        self._metrics.forward(clear_no_need_grad=True)
        # No need to save internal states
        metrics = np.squeeze(self._metrics.d, axis=0) if batch_size == 1 else self._metrics.d
        return metrics, {}

    def _compute_influence_metrics(self, state_var: nn.Variable, action_var: nn.Variable) -> nn.Variable:
        # TODO: support tuple state
        assert isinstance(state_var, nn.Variable), "Tuple states are not supported yet."

        num_factors = self._q_function.num_factors

        # compute base gradient
        # (B, A)
        action_var.need_grad = True
        base_factored_q = self._q_function.factored_q(state_var, action_var)
        base_grads = RF.expand_dims(nn.grad([-NF.sum(base_factored_q)], [action_var])[0], axis=1)

        # expand batch to factors
        # (B, S) -> (B, N, S)
        expand_state = RF.repeat(RF.expand_dims(state_var, axis=1), num_factors, axis=1)
        # (B, A) -> (B, N, A)
        expand_action = RF.repeat(RF.expand_dims(action_var, axis=1), num_factors, axis=1)

        # flatten shapes
        # (B, N, S) -> (B * N, S)
        flat_state = NF.reshape(expand_state, [-1, *state_var.shape[1:]])
        # (B, N, A) -> (B * N, A)
        flat_action = NF.reshape(expand_action, [-1, *action_var.shape[1:]])
        flat_action.need_grad = True

        # create mask
        # (N, N)
        mask = 1.0 - NF.one_hot(RF.expand_dims(NF.arange(0, num_factors), axis=1), shape=(num_factors,))
        # (N, N) -> (B, N, N)
        expand_mask = RF.repeat(RF.expand_dims(mask, axis=0), self._batch_size, axis=0)
        # (B * N, N)
        flat_mask = NF.reshape(expand_mask, [-1, num_factors])

        # compute Q-values
        # (B * N, N)
        factored_q = self._q_function.factored_q(flat_state, flat_action)

        # compute action gradients
        # (B * N, A)
        grads = nn.grad([-NF.sum(flat_mask * factored_q)], [flat_action])[0]

        # compute relative influence
        # (B * N, A) -> (B, N, A)
        squared_diff = (base_grads - NF.reshape(grads, [self._batch_size, num_factors, -1])) ** 2
        # (B, N, A) -> (B, N)
        influence = NF.sum(squared_diff, axis=2) ** 0.5

        # normalize
        return influence / (NF.sum(influence, axis=1, keepdims=True) + 1e-5)
