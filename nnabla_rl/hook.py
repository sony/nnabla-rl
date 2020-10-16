from abc import ABCMeta, abstractmethod


class Hook(metaclass=ABCMeta):
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


def as_hook(timing=None, **kwargs):
    if timing is None:
        timing = Hook.timing

    def decorate(hook):
        def decorated_hook(algorithm):
            if algorithm.iteration_num % timing == 0:
                hook(algorithm)
        return decorated_hook
    return decorate
