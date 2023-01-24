# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnabla_rl.algorithm import Algorithm


class Hook(metaclass=ABCMeta):
    """Base class of hooks for Algorithm classes.

    Hook is called at every 'timing' iterations during the training.
    'timing' is specified at the beginning of the class instantiation.
    """

    _timing: int = 1

    def __init__(self, timing: int = 1000):
        self._timing = timing

    def __call__(self, algorithm: 'Algorithm'):
        if algorithm.iteration_num % self._timing != 0:
            return
        self.on_hook_called(algorithm)

    @abstractmethod
    def on_hook_called(self, algorithm: 'Algorithm'):
        """Called every "timing" iteration which is set on Hook's instance
        creation. Will run additional periodical operation (see each class'
        documentation) during the training.

        Args:
            algorithm (:py:class:`Algorithm <nnabla_rl.algorithm.Algorithm>`):
                Algorithm instance to perform additional operation.
        """
        raise NotImplementedError

    def setup(self, algorithm: 'Algorithm', total_iterations: int):
        """Called before the training starts.

        Args:
            algorithm (:py:class:`Algorithm <nnabla_rl.algorithm.Algorithm>`):
                Algorithm instance to perform additional operation.
            total_iteration (int): the number of total iterations.
        """
        pass

    def teardown(self, algorithm: 'Algorithm', total_iterations: int):
        """Called after the training ends.

        Args:
            algorithm (:py:class:`Algorithm <nnabla_rl.algorithm.Algorithm>`):
                Algorithm instance to perform additional operation.
            total_iteration (int): the number of total iterations.
        """
        pass
