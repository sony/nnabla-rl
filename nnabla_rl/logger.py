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

import logging
from contextlib import contextmanager

logger = logging.getLogger('nnabla_rl')
logger.disabled = False


@contextmanager
def enable_logging(level=logging.INFO):
    return _switch_logability(disabled=False, level=level)


@contextmanager
def disable_logging(level=logging.INFO):
    return _switch_logability(disabled=True, level=level)


def _switch_logability(disabled, level=logging.INFO):
    global logger
    previous_level = logger.level
    previous_status = logger.disabled
    try:
        logger.disabled = disabled
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(previous_level)
        logger.disabled = previous_status
