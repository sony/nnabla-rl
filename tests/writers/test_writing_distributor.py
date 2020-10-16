import numpy as np

from unittest import mock

from nnabla_rl.writers.writing_distributor import WritingDistributor


class TestWritingDistributor():
    def test_write_scalar(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        distributor.write_scalar(1, {'test': 100})

        for writer in writers:
            writer.write_scalar.assert_called_once()

    def test_write_histogram(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        distributor.write_histogram(1, {'test': [100, 100]})

        for writer in writers:
            writer.write_histogram.assert_called_once()

    def test_write_image(self):
        writers = [self._new_mock_writer() for _ in range(10)]

        distributor = WritingDistributor(writers)
        image = np.empty(shape=(3, 10, 10))
        distributor.write_image(1, {'test': image})

        for writer in writers:
            writer.write_image.assert_called_once()

    def _new_mock_writer(self):
        return mock.MagicMock()
