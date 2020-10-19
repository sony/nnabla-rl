from abc import ABCMeta, abstractmethod

from nnabla_rl.models.model import Model


def preprocess_state(function):
    def wrapped(self, s):
        if self._state_preprocessor is not None:
            processed = self._state_preprocessor.process(s)
            return function(self, processed)
        else:
            raise NotImplementedError("decorated with preprecess_state but preprocessor is not set")
    return wrapped


class VFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name):
        super(VFunction, self).__init__(scope_name)
        self._state_preprocessor = None

    def set_state_preprocessor(self, preprocessor):
        self._state_preprocessor = preprocessor

    def __call__(self, s):
        raise NotImplementedError

    @abstractmethod
    def v(self, s):
        raise NotImplementedError
