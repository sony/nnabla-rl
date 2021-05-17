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

import nnabla_rl as rl
from nnabla_rl.algorithm import Algorithm, AlgorithmConfig, eval_api
from nnabla_rl.logger import logger


class DummyConfig(AlgorithmConfig):
    pass


class Dummy(Algorithm):
    '''
    This algorithm does nothing. Just used for understanding the concept of the framework.
    '''

    def __init__(self, env_or_env_info, config=DummyConfig()):
        super(Dummy, self).__init__(env_or_env_info, config=config)
        self._action_space = self._env_info.action_space

    @eval_api
    def compute_eval_action(self, state):
        return self._action_space.sample()

    def _build_evaluation_graph(self):
        assert rl.is_eval_scope()

    def _before_training_start(self, env_or_buffer):
        logger.debug("Before training start!! Write your algorithm's initializations here.")

    def _run_online_training_iteration(self, env):
        logger.debug("Running online training loop. Iteartion: {}".format(
            self.iteration_num))

    def _run_offline_training_iteration(self, buffer):
        logger.debug("Running offline training loop. Iteartion: {}".format(
            self.iteration_num))

    def _after_training_finish(self, env_or_buffer):
        logger.debug("Training finished. Do your algorithm's finalizations here.")

    def _models(self):
        return {}

    def _solvers(self):
        return {}
