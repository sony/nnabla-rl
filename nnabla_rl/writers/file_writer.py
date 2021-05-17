# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import numpy as np

import nnabla_rl.utils.files as files
from nnabla_rl.writer import Writer


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

        self._create_file_if_not_exists(outfile, out_scalar.keys())

        with open(outfile, 'a') as f:
            np.savetxt(f, [list(out_scalar.values())],
                       fmt=['%i'] + [self._fmt] * len_scalar,
                       delimiter='\t')

    def write_histogram(self, iteration_num, histogram):
        outfile = self._outdir / (self._file_prefix + '_histogram.tsv')

        self._create_file_if_not_exists(outfile, ['iteration(key)', 'values'])

        with open(outfile, 'a') as f:
            for key, values in histogram.items():
                np.savetxt(f, [[iteration_num] + [*values]],
                           fmt=[f'%i ({key})'] + [self._fmt] * len(values),
                           delimiter='\t')

    def write_image(self, iteration_num, image):
        pass

    def _create_file_if_not_exists(self, outfile, header_keys):
        if not outfile.exists():
            outfile.touch()
            self._write_file_header(outfile, header_keys)

    def _write_file_header(self, filepath, keys):
        with open(filepath, 'w+') as f:
            np.savetxt(f, [list(keys)], fmt='%s', delimiter='\t')
