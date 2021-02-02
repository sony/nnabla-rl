import gym

import numpy as np


class EpisodicEnv(gym.Wrapper):
    def __init__(self, env, min_episode_length=10):
        super(EpisodicEnv, self).__init__(env)
        self._min_episode_length = min_episode_length
        self._episode_length = 0

    def reset(self):
        self._episode_length = 0
        return self.env.reset()

    def step(self, action):
        self._episode_length += 1
        s_next, r, done, info = self.env.step(action)

        if self._episode_length > self._min_episode_length:
            done = np.random.randint(10) == 0
        return s_next, r, done, info


def generate_dummy_experiences(env, experience_num):
    experiences = []
    for _ in range(experience_num):
        state = env.reset()
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = env.action_space.sample()
            action = np.reshape(action, newshape=(1, ))
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        experience = (state, action, [reward], [1.0 - done], next_state)
        experiences.append(experience)
    return experiences


def is_same_parameter_id_and_key(param1, param2):
    ''' Compare two parameters have same ids and keys.
        Note this function does not check order of items.

    Args:
        param1 (Dict): parameters
        param2 (Dict): parameters
    Returns:
        bool: Have same parameters or not
    '''
    assert len(param1) == len(param2)

    for key1, value1 in param1.items():
        if key1 not in param2.keys():
            return False
        if value1 not in param2.values():
            return False

    return True
