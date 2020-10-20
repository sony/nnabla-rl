from nnabla_rl.replay_buffer import ReplayBuffer


class DecorableReplayBuffer(ReplayBuffer):
    '''Buffer which can decorate the experience with external decoration function

    This buffer enables decorating the experience before the item is used for building the batch.
    Decoration function will be called when __getitem__ is called.
    You can use this buffer to augment the data or add noise to the experience.
    '''

    def __init__(self, capacity, decor_fun):
        super(DecorableReplayBuffer, self).__init__(capacity=capacity)
        self._decor_fun = decor_fun

    def __getitem__(self, item):
        experience = self._buffer[item]
        return self._decor_fun(experience)
