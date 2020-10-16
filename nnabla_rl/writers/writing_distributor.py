import pathlib

import numpy as np

from nnabla_rl.writer import Writer


class WritingDistributor(Writer):
    def __init__(self, writers):
        super(WritingDistributor, self).__init__()
        self._writers = writers

    def write_scalar(self, iteration_num, scalar):
        for writer in self._writers:
            writer.write_scalar(iteration_num, scalar)

    def write_histogram(self, iteration_num, histogram):
        for writer in self._writers:
            writer.write_histogram(iteration_num, histogram)

    def write_image(self, iteration_num, image):
        for writer in self._writers:
            writer.write_image(iteration_num, image)

    def _write_file_header(self, filepath, keys):
        with open(filepath, 'w+') as f:
            np.savetxt(f, [list(keys)], fmt='%s', delimiter='\t')
