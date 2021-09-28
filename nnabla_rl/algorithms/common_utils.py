# Copyright 2020,2021 Sony Corporation.
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

from typing import Sequence, Tuple, Union

import numpy as np

import nnabla as nn
from nnabla_rl.models import DeterministicPolicy, Model, Policy, QFunction, RewardFunction, StochasticPolicy, VFunction
from nnabla_rl.preprocessors import Preprocessor
from nnabla_rl.typing import Experience
from nnabla_rl.utils.data import set_data_to_variable
from nnabla_rl.utils.misc import create_variable


def compute_v_target_and_advantage(v_function: VFunction,
                                   experiences: Sequence[Experience],
                                   gamma: float = 0.99,
                                   lmb: float = 0.97) -> Tuple[np.ndarray, np.ndarray]:
    ''' Compute value target and advantage by using Generalized Advantage Estimation (GAE)

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
    '''
    assert isinstance(v_function, VFunction), "Invalid v_function"

    def get_shape(state):
        if isinstance(state, tuple):
            return tuple(s.shape for s in state)
        else:
            return state.shape

    T = len(experiences)
    v_targets: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantages: np.ndarray = np.empty(shape=(T, 1), dtype=np.float32)
    advantage = 0.

    v_current = None
    v_next = None
    s_var = create_variable(1, get_shape(experiences[0][0]))
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

    def deepcopy(self, new_scope_name: str) -> Model:
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedVFunction)
        copied._v_function._scope_name = new_scope_name
        return copied


class _StatePreprocessedPolicy(Policy):
    _policy: Union[DeterministicPolicy, StochasticPolicy]
    _preprocessor: Preprocessor

    def __init__(self, policy: Union[DeterministicPolicy, StochasticPolicy], preprocessor: Preprocessor):
        super(_StatePreprocessedPolicy, self).__init__(policy.scope_name)
        self._policy = policy
        self._preprocessor = preprocessor

    def pi(self, s: nn.Variable):
        preprocessed_state = self._preprocessor.process(s)
        return self._policy.pi(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> Model:
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedPolicy)
        copied._policy._scope_name = new_scope_name
        return copied


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

    def deepcopy(self, new_scope_name: str) -> Model:
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedRewardFunction)
        copied._reward_function._scope_name = new_scope_name
        return copied


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

    def deepcopy(self, new_scope_name: str) -> Model:
        copied = super().deepcopy(new_scope_name=new_scope_name)
        assert isinstance(copied,  _StatePreprocessedQFunction)
        copied._q_function._scope_name = new_scope_name
        return copied
