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


import nnabla.functions as NF
from nnabla_rl.models.model import Model
from nnabla_rl.preprocessors.preprocessor import Preprocessor
from nnabla_rl.preprocessors.running_mean_normalizer import RunningMeanNormalizer
from nnabla_rl.utils.misc import create_variable


class HERMeanNormalizer(RunningMeanNormalizer):
    def __init__(self, scope_name, shape, epsilon=1e-8, value_clip=None):
        super(HERMeanNormalizer, self).__init__(scope_name, shape, epsilon, value_clip)

        self._epsilon = create_variable(batch_size=1, shape=shape)
        self._epsilon.d = epsilon

    def process(self, x):
        assert 0 < self._count.d
        std = NF.maximum2(self._var ** 0.5, self._epsilon)
        normalized = (x - self._mean) / std
        if self._value_clip is not None:
            normalized = NF.clip_by_value(normalized, min=self._value_clip[0], max=self._value_clip[1])
        normalized.need_grad = False
        return normalized


class HERPreprocessor(Preprocessor, Model):
    def __init__(self, scope_name, shape, epsilon=1e-8, value_clip=None):
        super(HERPreprocessor, self).__init__(scope_name)

        observation_shape, goal_shape, _ = shape
        self._observation_preprocessor = HERMeanNormalizer(scope_name=f'{scope_name}/observation',
                                                           shape=observation_shape,
                                                           epsilon=epsilon,
                                                           value_clip=value_clip)
        self._goal_preprocessor = HERMeanNormalizer(scope_name=f'{scope_name}/goal',
                                                    shape=goal_shape,
                                                    epsilon=epsilon,
                                                    value_clip=value_clip)

    def process(self, x):
        observation, goal, achived_goal = x
        normalized_observation = self._observation_preprocessor.process(observation)
        normalized_goal = self._goal_preprocessor.process(goal)
        normalized = (normalized_observation, normalized_goal, achived_goal)
        return normalized

    def update(self, data):
        observation, goal, _ = data
        self._observation_preprocessor.update(observation)
        self._goal_preprocessor.update(goal)
