import numpy as np

prng = np.random.RandomState()


def seed(seed=None):
    global prng
    prng = np.random.RandomState(seed=seed)
