import pathlib

import json

import pickle

from nnabla_rl.algorithm import Algorithm
import nnabla_rl.algorithms as A
from nnabla_rl.environments.environment_info import EnvironmentInfo
import nnabla_rl.utils.files as files

_TRAINING_INFO_FILENAME = 'training_info.json'
_ENV_INFO_FILENAME = 'env_info.pickle'

_KEY_ALGORITHM_NAME = 'algorithm_name'
_KEY_ALGORITHM_CLASS_NAME = 'algorithm_class_name'
_KEY_ALGORITHM_PARAMS = 'algorithm_params'
_KEY_ITERATION_NUM = 'iteration_num'
_KEY_MODELS = 'models'
_KEY_SOLVERS = 'solvers'


def save_snapshot(path, algorithm):
    """Save training snapshot to file

    Args:
      path(str or pathlib.Path): Path to the snapshot saved dir
      algorithm(nnabla_rl.Algorithm): Algorithm object to save the snapshot

    Returns: pathlib.Path
      File path where the snapshot is saved to
    """
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
    """Load training snapshot from file

    Args:
      path(str or pathlib.Path): Path to the snapshot saved dir
      algorithm_kwargs(dictionary): parameters passed to the constructor of loaded algorithm class

    Returns: nnabla_rl.Algorithm
      Algorithm with parameters and settings loaded from file
    """
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
    (algorithm_klass, params_klass) = A.get_class_of(algorithm_name)

    if kwargs.get('params', None) is None:
        saved_params = training_info[_KEY_ALGORITHM_PARAMS]
        kwargs['params'] = params_klass(**saved_params)
    algorithm = algorithm_klass(env_info, **kwargs)
    algorithm._iteration_num = training_info[_KEY_ITERATION_NUM]
    return algorithm


def _create_training_info(algorithm):
    training_info = {}
    training_info[_KEY_ALGORITHM_NAME] = algorithm.__name__
    training_info[_KEY_ALGORITHM_CLASS_NAME] = algorithm.__class__.__name__
    training_info[_KEY_ALGORITHM_PARAMS] = algorithm._params.to_dict()
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
    for model_name, model in algorithm._models().items():
        filename = model_name + '.h5'
        filepath = path / filename
        model.save_parameters(filepath)


def _load_network_parameters(path, algorithm):
    for model_name, model in algorithm._models().items():
        filename = model_name + '.h5'
        filepath = path / filename
        model.load_parameters(filepath)


def _save_solver_states(path, algorithm):
    for solver_name, solver in algorithm._solvers().items():
        filename = solver_name + '.h5'
        filepath = path / filename
        solver.save_states(filepath)


def _load_solver_states(path, algorithm):
    for solver_name, solver in algorithm._solvers().items():
        filename = solver_name + '.h5'
        filepath = path / filename
        solver.load_states(filepath)
