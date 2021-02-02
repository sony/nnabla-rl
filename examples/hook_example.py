import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.hook import Hook
from nnabla_rl.replay_buffer import ReplayBuffer


class PrintHello(Hook):
    def __init__(self):
        super().__init__(timing=1)

    def on_hook_called(self, algorithm):
        print('hello!!')


class PrintOnlyEvenIteraion(Hook):
    def __init__(self):
        super().__init__(timing=2)

    def on_hook_called(self, algorithm):
        print('even iteration -> {}'.format(algorithm.iteration_num))


def main():
    dummy_env = E.DummyContinuous()
    empty_buffer = ReplayBuffer()

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[PrintHello()])
    dummy_algorithm.train(empty_buffer, total_iterations=10)

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[PrintHello(), PrintOnlyEvenIteraion()])
    dummy_algorithm.train(empty_buffer, total_iterations=10)


if __name__ == "__main__":
    main()
