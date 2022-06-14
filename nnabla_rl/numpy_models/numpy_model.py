# Copyright 2022 Sony Group Corporation.
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
from typing import Union


class NumpyModel(object):
    def save_parameters(self, filepath: Union[str, pathlib.Path]) -> None:
        '''save_parameters
        Save model parameters to given filepath.
        Args:
            filepath (str or pathlib.Path): paramter file path
        '''
        raise NotImplementedError

    def load_parameters(self, filepath: Union[str, pathlib.Path]) -> None:
        '''load_parameters
        Load model parameters from given filepath.
        Args:
            filepath (str or pathlib.Path): paramter file path
        '''
        raise NotImplementedError

    def support_batch(self) -> bool:
        '''support_batch
        Check whether the model supports batched inputs or not.

        Returns:
            bool: True if supports batched input otherwise False.
        '''
        return False
