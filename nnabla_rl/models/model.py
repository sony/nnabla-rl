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

import copy
import pathlib
from typing import Dict, Union

import nnabla as nn
from nnabla_rl.logger import logger


class Model(object):
    """Model Class

    Args:
        scope_name (str): the scope name of model
    """

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _scope_name: str

    def __init__(self, scope_name: str):
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        '''scope_name
        Get scope name of this model.

        Returns:
            scope_name (str): scope name of the model
        '''
        return self._scope_name

    def get_parameters(self, grad_only: bool = True) -> Dict[str, nn.Variable]:
        '''get_parameters
        Retrive parameters associated with this model

        Args:
            grad_only (bool): Retrive parameters only with need_grad = True. Defaults to True.

        Returns:
            parameters (OrderedDict): Parameter map.
        '''
        with nn.parameter_scope(self.scope_name):
            parameters: Dict[str, nn.Variable] = nn.get_parameters(grad_only=grad_only)
            return parameters

    def save_parameters(self, filepath: Union[str, pathlib.Path]) -> None:
        '''save_parameters
        Save model parameters to given filepath.

        Args:
            filepath (str or pathlib.Path): paramter file path
        '''
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.save_parameters(path=filepath)

    def load_parameters(self, filepath: Union[str, pathlib.Path]) -> None:
        '''load_parameters
        Load model parameters from given filepath.

        Args:
            filepath (str or pathlib.Path): paramter file path
        '''
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.load_parameters(path=filepath)

    def deepcopy(self, new_scope_name: str) -> 'Model':
        '''deepcopy
        Create a copy of the model. All the model parameter's (if exist) associated with will be copied.

        Args:
            new_scope_name (str): scope_name of parameters for newly created model

        Returns:
            Model: copied model

        Raises:
            ValueError: Given scope name is same as the model or already exists.
        '''
        assert new_scope_name != self._scope_name, 'Can not use same scope_name!'
        copied = copy.deepcopy(self)
        copied._scope_name = new_scope_name
        # copy current parameter if is already created
        params = self.get_parameters(grad_only=False)
        with nn.parameter_scope(new_scope_name):
            for param_name, param in params.items():
                if nn.parameter.get_parameter(param_name) is not None:
                    raise RuntimeError(f'Model with scope_name: {new_scope_name} already exists!!')
                logger.info(
                    f'copying param with name: {self.scope_name}/{param_name} ---> {new_scope_name}/{param_name}')
                nn.parameter.get_parameter_or_create(param_name, shape=param.shape, initializer=param.d)
        return copied
