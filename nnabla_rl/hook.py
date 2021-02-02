from abc import ABCMeta, abstractmethod


class Hook(metaclass=ABCMeta):
    '''
    Base class of hooks for Algorithm classes.

    Hook is called at every 'timing' iterations during the training.
    'timing' is specified at the beginning of the class instantiation.
    '''

    timing = 1

    def __init__(self, timing=1000):
        self._timing = timing

    def __call__(self, algorithm):
        if algorithm.iteration_num % self._timing != 0:
            return
        self.on_hook_called(algorithm)

    @abstractmethod
    def on_hook_called(self, algorithm):
        raise NotImplementedError
