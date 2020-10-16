import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.hook import as_hook
from nnabla_rl.replay_buffer import ReplayBuffer


@as_hook(timing=1)
def print_hello(algorithm):
    print('hello!!')


@as_hook(timing=2)
def print_only_on_even_iteration(algorithm):
    print('even iteration -> {}'.format(algorithm.iteration_num))


def main():
    dummy_env = E.DummyContinuous()
    empty_buffer = ReplayBuffer()

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[print_hello])
    dummy_algorithm.train(empty_buffer, total_iterations=10)

    dummy_algorithm = A.Dummy(dummy_env)
    dummy_algorithm.set_hooks(hooks=[print_only_on_even_iteration])
    dummy_algorithm.train(empty_buffer, total_iterations=10)


if __name__ == "__main__":
    main()
