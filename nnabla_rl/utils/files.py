# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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

import os


def file_exists(path):
    '''
    Check file existence on given path
    Parameters
    -------
    path : string
        Path of the file to check existence
    Returns
    -------
    file_existence : bool
        True if file exists otherwise False
    '''
    return os.path.exists(path)


def create_dir_if_not_exist(outdir):
    '''
    Check directory existence and creates new directory if not exist
    Parameters
    -------
    outdir : string
        Path of the file to create directory
    Raises
    ------
    RuntimeError
        File exists in outdir but it is not a directory
    '''
    if file_exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def read_text_from_file(file_path):
    '''
    Read given file as text
    Parameters
    -------
    file_path : string
        Path of the file to read data
    Returns
    -------
    data: string
        Text read from the file
    '''
    with open(file_path, 'r') as f:
        return f.read()


def write_text_to_file(file_path, data):
    '''
    Write given text data to file
    Parameters
    -------
    file_path : string
        Path of the file to write data
    data: string
        Text to write to the file
    '''
    with open(file_path, 'w') as f:
        f.write(data)
