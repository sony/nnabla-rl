from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, x):
        raise NotImplementedError

    def update(self, data):
        pass
