import nnabla as nn
import numpy as np

from .. import models as M


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
    assert isinstance(v_function, M.VFunction), "Invalid v_function"
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
