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

import json
import pathlib
import pickle
import warnings

import nnabla_rl.algorithms as A
import nnabla_rl.utils.files as files
from nnabla_rl.algorithm import Algorithm

_TRAINING_INFO_FILENAME = 'training_info.json'
_ENV_INFO_FILENAME = 'env_info.pickle'

_KEY_ALGORITHM_NAME = 'algorithm_name'
_KEY_ALGORITHM_CLASS_NAME = 'algorithm_class_name'
_KEY_ALGORITHM_CONFIG = 'algorithm_config'
_KEY_ITERATION_NUM = 'iteration_num'
_KEY_MODELS = 'models'
_KEY_SOLVERS = 'solvers'


def save_snapshot(path, algorithm):
    '''Save training snapshot to file

    Args:
      path(str or pathlib.Path): Path to the snapshot saved dir
      algorithm(nnabla_rl.Algorithm): Algorithm object to save the snapshot

    Returns: pathlib.Path
      File path where the snapshot is saved to
    '''
    assert isinstance(algorithm, Algorithm)
    if isinstance(path, str):
        path = pathlib.Path(path)
    dirname = 'iteration-' + str(algorithm.iteration_num)
    outdir = path / dirname
    files.create_dir_if_not_exist(outdir=outdir)

    training_info = _create_training_info(algorithm)
    _save_training_info(outdir, training_info)
    _save_env_info(outdir, algorithm)
    _save_network_parameters(outdir, algorithm)
    _save_solver_states(outdir, algorithm)

    return outdir


def load_snapshot(path,
                  algorithm_kwargs={}):
    '''Load training snapshot from file

    Args:
      path(str or pathlib.Path): Path to the snapshot saved dir
      algorithm_kwargs(dictionary): parameters passed to the constructor of loaded algorithm class

    Returns: nnabla_rl.Algorithm
      Algorithm with parameters and settings loaded from file
    '''
    if isinstance(path, str):
        path = pathlib.Path(path)
    training_info = _load_training_info(path)
    env_info = _load_env_info(path)
    algorithm = _instantiate_algorithm_from_training_info(
        training_info, env_info, **algorithm_kwargs)
    _load_network_parameters(path, algorithm)
    _load_solver_states(path, algorithm)
    return algorithm


def _instantiate_algorithm_from_training_info(training_info, env_info, **kwargs):
    algorithm_name = training_info[_KEY_ALGORITHM_CLASS_NAME]
    (algorithm_klass, config_klass) = A.get_class_of(algorithm_name)

    if kwargs.get('config', None) is None:
        saved_config = training_info[_KEY_ALGORITHM_CONFIG]
        kwargs['config'] = config_klass(**saved_config)
    algorithm = algorithm_klass(env_info, **kwargs)
    algorithm._iteration_num = training_info[_KEY_ITERATION_NUM]
    return algorithm


def _create_training_info(algorithm):
    training_info = {}
    training_info[_KEY_ALGORITHM_NAME] = algorithm.__name__
    training_info[_KEY_ALGORITHM_CLASS_NAME] = algorithm.__class__.__name__
    training_info[_KEY_ALGORITHM_CONFIG] = algorithm._config.to_dict()
    training_info[_KEY_ITERATION_NUM] = algorithm.iteration_num
    training_info[_KEY_MODELS] = list(algorithm._models().keys())
    training_info[_KEY_SOLVERS] = list(algorithm._solvers().keys())

    return training_info


def _save_env_info(path, algorithm):
    filepath = path / _ENV_INFO_FILENAME
    with open(filepath, 'wb+') as outfile:
        pickle.dump(algorithm._env_info, outfile)


def _load_env_info(path):
    filepath = path / _ENV_INFO_FILENAME
    with open(filepath, 'rb') as infile:
        return pickle.load(infile)


def _save_training_info(path, training_info):
    filepath = path / _TRAINING_INFO_FILENAME
    with open(filepath, 'w+') as outfile:
        json.dump(training_info, outfile)


def _load_training_info(path):
    filepath = path / _TRAINING_INFO_FILENAME
    with open(filepath, 'r') as infile:
        training_info = json.load(infile)
    return training_info


def _save_network_parameters(path, algorithm):
    for scope_name, model in algorithm._models().items():
        filename = scope_name + '.h5'
        filepath = path / filename
        model.save_parameters(filepath)


def _load_network_parameters(path, algorithm):
    for scope_name, model in algorithm._models().items():
        filename = scope_name + '.h5'
        filepath = path / filename
        model.load_parameters(filepath)


def _save_solver_states(path, algorithm):
    for scope_name, solver in algorithm._solvers().items():
        filename = scope_name + '_solver' + '.h5'
        filepath = path / filename
        solver.save_states(filepath)


def _load_solver_states(path, algorithm):
    models = algorithm._models()
    for scope_name, solver in algorithm._solvers().items():
        filename = scope_name + '_solver' + '.h5'
        filepath = path / filename
        if not filepath.exists():
            warnings.warn(f"No solver file found in: {filepath}. Ommitting...")
            continue
        model = models[scope_name]
        solver.set_parameters(model.get_parameters())
        solver.load_states(filepath)
