import pathlib

import numpy as np

from nnabla_rl.writer import Writer
import nnabla_rl.utils.files as files


class FileWriter(Writer):
    def __init__(self, outdir, file_prefix):
        super(FileWriter, self).__init__()
        if isinstance(outdir, str):
            outdir = pathlib.Path(outdir)
        self._outdir = outdir
        files.create_dir_if_not_exist(outdir=outdir)
        self._file_prefix = file_prefix
        self._fmt = '%.3f'

    def write_scalar(self, iteration_num, scalar):
        outfile = self._outdir / (self._file_prefix + '_scalar.tsv')

        len_scalar = len(scalar.values())
        out_scalar = {}
        out_scalar['iteration'] = iteration_num
        out_scalar.update(scalar)

        if not outfile.exists():
            outfile.touch()
            self._write_file_header(outfile, out_scalar.keys())

        with open(outfile, 'a') as f:
            np.savetxt(f, [list(out_scalar.values())],
                       fmt=['%i'] + [self._fmt] * len_scalar,
                       delimiter='\t')

    def write_histogram(self, iteration_num, histogram):
        pass

    def write_image(self, iteration_num, image):
        pass

    def _write_file_header(self, filepath, keys):
        with open(filepath, 'w+') as f:
            np.savetxt(f, [list(keys)], fmt='%s', delimiter='\t')
