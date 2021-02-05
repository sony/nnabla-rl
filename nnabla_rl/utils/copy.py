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

def copy_network_parameters(origin_params, target_params, tau=1.0):

    if not ((0.0 <= tau) & (tau <= 1.0)):
        raise ValueError('tau must lie between [0.0, 1.0]')

    for key in target_params.keys():
        target_params[key].data.copy_from(
            origin_params[key].data * tau
            + target_params[key].data * (1 - tau))
