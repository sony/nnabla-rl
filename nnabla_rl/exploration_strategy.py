from abc import ABCMeta, abstractmethod


class ExplorationStrategy(metaclass=ABCMeta):
    @abstractmethod
    def select_action(self, step, state):
        pass
