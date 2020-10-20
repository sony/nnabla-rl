from unittest import mock

from nnabla_rl.hooks import IterationStateHook
from nnabla_rl.writer import Writer


class TestIterationStateHook():
    def test_call(self):
        dummy_algorithm = mock.MagicMock()

        test_latest_iteration_state = {}
        test_latest_iteration_state['scalar'] = {}
        test_latest_iteration_state['histogram'] = {}
        test_latest_iteration_state['image'] = {}

        dummy_algorithm.iteration_num = 1
        dummy_algorithm.latest_iteration_state = test_latest_iteration_state

        writer = Writer()
        writer.write_scalar = mock.MagicMock()
        writer.write_histogram = mock.MagicMock()
        writer.write_image = mock.MagicMock()

        hook = IterationStateHook(writer=writer, timing=1)

        hook(dummy_algorithm)

        writer.write_scalar.assert_called_once_with(dummy_algorithm.iteration_num,
                                                    test_latest_iteration_state['scalar'])
        writer.write_histogram.assert_called_once_with(dummy_algorithm.iteration_num,
                                                       test_latest_iteration_state['histogram'])
        writer.write_image.assert_called_once_with(dummy_algorithm.iteration_num,
                                                   test_latest_iteration_state['image'])
