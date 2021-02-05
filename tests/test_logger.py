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

import pytest

import nnabla_rl.logger as logger


class TestLogger(object):
    def test_enable_logging(self):
        # disable logging
        logger.logger.disabled = True
        assert logger.logger.disabled
        with logger.enable_logging():
            assert not logger.logger.disabled
        assert logger.logger.disabled

    def test_disable_logging(self):
        logger.logger.disabled = False
        assert not logger.logger.disabled
        with logger.disable_logging():
            assert logger.logger.disabled
        assert not logger.logger.disabled


if __name__ == "__main__":
    pytest.main()
