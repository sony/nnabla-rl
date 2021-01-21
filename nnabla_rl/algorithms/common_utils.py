import nnabla as nn
import numpy as np

from nnabla_rl.models import VFunction, StochasticPolicy, Model
from nnabla_rl.preprocessors import Preprocessor


def compute_v_target_and_advantage(v_function, experiences, gamma=0.99, lmb=0.97):
    """ Compute value target and advantage by using Generalized Advantage Estimation (GAE)

    Args:
        v_function (M.VFunction): value function
        experiences (list): list of experience.
            experience should have [state_current, action, reward, non_terminal, state_next]
        gamma (float): discount rate
        lmb (float): lambda
        preprocess_func (callable): preprocess function of states
    Returns:
        v_target (numpy.ndarray): target of value
        advantage (numpy.ndarray): advantage
    Ref:
        https://arxiv.org/pdf/1506.02438.pdf
    """
    assert isinstance(v_function, VFunction), "Invalid v_function"
    T = len(experiences)
    v_targets = []
    advantages = []
    advantage = 0.

    v_current = None
    v_next = None
    state_shape = experiences[0][0].shape  # get state shape
    s_var = nn.Variable((1, *state_shape))
    v = v_function.v(s_var)  # build graph

    for t in reversed(range(T)):
        s_current, _, r, non_terminal, s_next, *_ = experiences[t]

        # predict current v
        s_var.d = s_current
        v.forward()
        v_current = np.squeeze(v.d)

        if v_next is None:
            s_var.d = s_next
            v.forward()
            v_next = np.squeeze(v.d)

        delta = r + gamma * non_terminal * v_next - v_current
        advantage = np.float32(
            delta + gamma * lmb * non_terminal * advantage)
        # A = Q - V, V = E[Q] -> v_target = A + V
        v_target = advantage + v_current

        v_targets.insert(0, v_target)
        advantages.insert(0, advantage)

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
        copied._v_function._scope_name = new_scope_name
        return copied


class _StatePreprocessedPolicy(StochasticPolicy):
    _policy: StochasticPolicy
    _preprocessor: Preprocessor

    def __init__(self, policy: StochasticPolicy, preprocessor: Preprocessor):
        super(_StatePreprocessedPolicy, self).__init__(policy.scope_name)
        self._policy = policy
        self._preprocessor = preprocessor

    def pi(self, s: nn.Variable):
        preprocessed_state = self._preprocessor.process(s)
        return self._policy.pi(preprocessed_state)

    def deepcopy(self, new_scope_name: str) -> Model:
        copied = super().deepcopy(new_scope_name=new_scope_name)
        copied._policy._scope_name = new_scope_name
        return copied
