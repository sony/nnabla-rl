import numpy as np

import pathlib
import tempfile
import os

from unittest import mock

from nnabla_rl.writers.file_writer import FileWriter


class TestFileWriter():
    def test_write_scalar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_returns = np.arange(5)
            test_results = {}
            test_results['mean'] = np.mean(test_returns)
            test_results['std_dev'] = np.std(test_returns)
            test_results['min'] = np.min(test_returns)
            test_results['max'] = np.max(test_returns)
            test_results['median'] = np.median(test_returns)

            writer = FileWriter(
                outdir=tmpdir, file_prefix='evaluation_results')
            writer.write_scalar(1, test_results)

            file_path = \
                os.path.join(tmpdir, 'evaluation_results_scalar.tsv')
            this_file_dir = os.path.dirname(__file__)
            test_file_dir = this_file_dir.replace('tests', 'test_resources')
            test_file_path = \
                os.path.join(test_file_dir, 'evaluation_results_scalar.tsv')
            self._check_same_tsv_file(file_path, test_file_path)

    def _check_same_tsv_file(self, file_path1, file_path2):
        # check each line
        with open(file_path1, mode='rt') as data_1, \
                open(file_path2, mode='rt') as data_2:
            for d_1, d_2 in zip(data_1, data_2):
                assert d_1 == d_2
