# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

from dataclasses import dataclass, asdict


@dataclass
class Parameter():
    def __post_init__(self):
        pass

    def to_dict(self):
        return asdict(self)

    def _assert_positive(self, param, var_name):
        if param <= 0:
            raise ValueError('{} must be positive'.format(var_name))

    def _assert_positive_or_zero(self, param, var_name):
        if param < 0:
            raise ValueError('{} must be positive'.format(var_name))

    def _assert_negative(self, param, var_name):
        if 0 <= param:
            raise ValueError('{} must be negative'.format(var_name))

    def _assert_negative_or_zero(self, param, var_name):
        if 0 < param:
            raise ValueError('{} must be positive'.format(var_name))

    def _assert_between(self, param, low, high, var_name):
        if not (low <= param and param <= high):
            raise ValueError(
                '{} must lie between [{}, {}]'.format(var_name, low, high))

    def _assert_one_of(self, param, choices, var_name):
        if param not in choices:
            raise ValueError(f'{var_name} is not available. Available choices: {choices}')

    def _assert_ascending_order(self, param, var_name):
        ascending = all(param[i] <= param[i+1] for i in range(len(param)-1))
        if not ascending:
            raise ValueError(f'{var_name} is not in ascending order!: {param}')

    def _assert_descending_order(self, param, var_name):
        descending = all(param[i] >= param[i+1] for i in range(len(param)-1))
        if not descending:
            raise ValueError(f'{var_name} is not in descending order!: {param}')

    def _assert_smaller_than(self, param, ref_value, var_name):
        if param > ref_value:
            raise ValueError(f'{var_name} is not in smaller than reference value!: {param} > {ref_value}')

    def _assert_length(self, param, expected_length, var_name):
        if len(param) != expected_length:
            raise ValueError(f'{var_name} length is not {expected_length}')
