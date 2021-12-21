# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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
from typing import Dict, Optional, Tuple, TypeVar, Union

import nnabla as nn
from nnabla_rl.logger import logger

T = TypeVar('T', bound='Model')


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

    def is_recurrent(self) -> bool:
        '''is_recurrent
        Check whether the model uses recurrent network component or not.
        Model which use LSTM, GRU and/or any other recurrent network component must return True.
        Returns:
            bool: True if the model uses recurrent network component. Otherwise False.
        '''
        return False

    def internal_state_shapes(self) -> Dict[str, Tuple[int, ...]]:
        '''internal_state_shapes
        Return internal state shape as tuple of ints for each internal state (excluding the batch_size).
        This method will be called by
        (:py:class:`RNNModelTrainer <nnabla_rl.model_trainers.model_trainer.RNNModelTrainer>`) and its subclasses
        to setup training variables.
        Model which use LSTM, GRU and/or any other recurrent network component must implement this method.

        Returns:
            Dict[str, Tuple[int, ...]]: internal state shapes. key is the name of each internal state.
        '''
        raise NotImplementedError

    def set_internal_states(self, states: Optional[Dict[str, nn.Variable]] = None):
        '''set_internal states
        Set the internal state variable of rnn cell to given state.
        Model which use LSTM, GRU and/or any other recurrent network component must implement this method.
        Args:
            states (None or Dict[str, nn.Variable]): If None, reset all internal state to zero.
            If state is provided, set the provided state as internal state.
        '''
        raise NotImplementedError

    def reset_internal_states(self):
        '''reset_internal states
        Set the internal state variable of rnn cell to given zero.
        '''
        self.set_internal_states(None)

    def get_internal_states(self) -> Dict[str, nn.Variable]:
        '''get_internal states
        Get the internal state variable of rnn cell.
        Model which use LSTM, GRU and/or any other recurrent network component must implement this method.

        Returns:
            Dict[str, nn.Variable]: Value of each internal state. key is the name of each internal state.
        '''
        raise NotImplementedError

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

    def deepcopy(self: T, new_scope_name: str) -> T:
        '''deepcopy
        Create a (deep) copy of the model. All the model parameter's (if exist) associated with will be copied and
        new_scope_name will be assigned.

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

    def shallowcopy(self: T) -> T:
        '''shallowcopy
        Create a (shallow) copy of the model.
        Unlike deepcopy, shallowcopy will KEEP sharing the original network parameter
        by using same scope_name as original model.
        However, all the class members will be (deep) copied to the new instance.
        Do NOT use this method unless you understand what this method does.

        Returns:
            Model: (shallow) copied model
        '''
        copied = copy.deepcopy(self)
        return copied
