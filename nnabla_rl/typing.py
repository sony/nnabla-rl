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

from typing import Any, Dict, Tuple, Union

import numpy as np

State = Union[np.ndarray, Tuple[np.ndarray, ...]]
# https://github.com/python/mypy/issues/7866
# FIXME: This is a workaround for avoiding mypy error about creating a type alias.
Action = Union[np.ndarray]
Reward = float
NonTerminal = float
NextState = Union[np.ndarray, Tuple[np.ndarray, ...]]
Info = Dict[str, Any]
Experience = Tuple[State, Action, Reward, NonTerminal, NextState, Info]
Shape = Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
TupledData = Tuple
