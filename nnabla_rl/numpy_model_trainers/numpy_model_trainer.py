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

from dataclasses import dataclass

from nnabla_rl.configuration import Configuration


@dataclass
class NumpyModelTrainerConfig(Configuration):
    '''Configuration class for ModelTrainer
    '''

    def __post_init__(self):
        super(NumpyModelTrainerConfig, self).__post_init__()


class NumpyModelTrainer(object):
    def __init__(self, config: NumpyModelTrainerConfig):
        self._config = config
