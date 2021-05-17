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

import pytest

from nnabla_rl.environment_explorer import _is_end_of_episode


class TestEnvironmentExplorer(object):
    @pytest.mark.parametrize("done", [True, False])
    @pytest.mark.parametrize("timelimit", [True, False])
    @pytest.mark.parametrize("timelimit_as_terminal", [True, False])
    def test_is_end_of_episode(self, done, timelimit, timelimit_as_terminal):
        end_of_episode = _is_end_of_episode(done, timelimit, timelimit_as_terminal)
        if not done:
            assert end_of_episode is False
        else:
            # All the case that done == True
            if timelimit and timelimit_as_terminal:
                assert end_of_episode is True
            elif timelimit and not timelimit_as_terminal:
                assert end_of_episode is False
            elif not timelimit:
                assert end_of_episode is True
            else:
                raise RuntimeError


if __name__ == '__main__':
    pytest.main()
