# Copyright 2021 Sony Corporation.
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

from dataclasses import asdict, dataclass


@dataclass
class Configuration():
    def __post_init__(self):
        pass

    def to_dict(self):
        return asdict(self)

    def _assert_positive(self, config, config_name):
        if config <= 0:
            raise ValueError('{} must be positive'.format(config_name))

    def _assert_positive_or_zero(self, config, config_name):
        if config < 0:
            raise ValueError('{} must be positive'.format(config_name))

    def _assert_negative(self, config, config_name):
        if 0 <= config:
            raise ValueError('{} must be negative'.format(config_name))

    def _assert_negative_or_zero(self, config, config_name):
        if 0 < config:
            raise ValueError('{} must be positive'.format(config_name))

    def _assert_between(self, config, low, high, config_name):
        if not (low <= config and config <= high):
            raise ValueError(
                '{} must lie between [{}, {}]'.format(config_name, low, high))

    def _assert_one_of(self, config, choices, config_name):
        if config not in choices:
            raise ValueError(f'{config_name} is not available. Available choices: {choices}')

    def _assert_ascending_order(self, config, config_name):
        ascending = all(config[i] <= config[i+1] for i in range(len(config)-1))
        if not ascending:
            raise ValueError(f'{config_name} is not in ascending order!: {config}')

    def _assert_descending_order(self, config, config_name):
        descending = all(config[i] >= config[i+1] for i in range(len(config)-1))
        if not descending:
            raise ValueError(f'{config_name} is not in descending order!: {config}')

    def _assert_smaller_than(self, config, ref_value, config_name):
        if config > ref_value:
            raise ValueError(f'{config_name} is not in smaller than reference value!: {config} > {ref_value}')

    def _assert_length(self, config, expected_length, config_name):
        if len(config) != expected_length:
            raise ValueError(f'{config_name} length is not {expected_length}')
