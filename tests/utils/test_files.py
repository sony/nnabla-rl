import pytest
import tempfile
import datetime
import os
from unittest import mock

from nnabla_rl.utils import files


class TestFiles(object):
    def test_file_exists(self):
        with mock.patch('os.path.exists', return_value=True) as _:
            test_file = "test"
            assert files.file_exists(test_file) is True

    def test_file_does_not_exist(self):
        with mock.patch('os.path.exists', return_value=False) as _:
            test_file = "test"
            assert files.file_exists(test_file) is False

    def test_create_dir_if_not_exist(self):
        with mock.patch('os.path.exists', return_value=False) as mock_exists, \
                mock.patch('os.makedirs') as mock_mkdirs:
            test_file = "test"
            files.create_dir_if_not_exist(test_file)

            mock_exists.assert_called_once()
            mock_mkdirs.assert_called_once()

    def test_create_dir_when_exists(self):
        with mock.patch('os.path.exists', return_value=True) as mock_exists, \
                mock.patch('os.makedirs') as mock_mkdirs:
            with mock.patch('os.path.isdir', return_value=True):
                test_file = "test"
                files.create_dir_if_not_exist(test_file)

                mock_exists.assert_called_once()
                mock_mkdirs.assert_not_called()

    def test_create_dir_when_target_is_not_directory(self):
        with mock.patch('os.path.exists', return_value=True) as mock_exists, \
                mock.patch('os.makedirs') as mock_mkdirs:
            with mock.patch('os.path.isdir', return_value=False):
                with pytest.raises(RuntimeError):
                    test_file = "test"
                    files.create_dir_if_not_exist(test_file)

                    mock_exists.assert_called_once()
                    mock_mkdirs.assert_not_called()

    def test_read_write_text_to_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            target_path = os.path.join(tempdir, 'test.txt')
            time_format = '%Y-%m-%d-%H%M%S.%f'
            test_text = datetime.datetime.now().strftime(time_format)
            files.write_text_to_file(target_path, test_text)

            assert files.file_exists(target_path)

            read_text = files.read_text_from_file(target_path)
            assert read_text == test_text


if __name__ == '__main__':
    pytest.main()
