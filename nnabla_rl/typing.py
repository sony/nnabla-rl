# Copyright 2021 Sony Corporation.
# Copyright 2021,2022,2023 Sony Group Corporation.
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

from inspect import signature
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

F = TypeVar('F', bound=Callable[..., Any])


State = Union[np.ndarray, Tuple[np.ndarray, ...]]
# https://github.com/python/mypy/issues/7866
# FIXME: This is a workaround for avoiding mypy error about creating a type alias.
Action = Union[np.ndarray]
Reward = Union[float, np.ndarray]
NonTerminal = float
NextState = Union[np.ndarray, Tuple[np.ndarray, ...]]
Info = Dict[str, Any]
Experience = Tuple[State, Action, Reward, NonTerminal, NextState, Info]
Shape = Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
TupledData = Tuple
Trajectory = Sequence[Experience]


try:
    from typing_extensions import Protocol
except ModuleNotFoundError:
    # User have not installed typing_extensions.
    # However, typing_extension is not necessary for running the library
    Protocol = object  # type: ignore


class ActionSelector(Protocol):
    def __call__(self, state: np.ndarray, *, begin_of_episode=False) -> Tuple[np.ndarray, Dict]: ...


def accepted_shapes(**shape_kwargs: Dict[str, Optional[Tuple[int]]]) -> Callable[[F], F]:
    """Accepted shape decorator. This decorator checks the argument shapes are
    the same as the expected shapes. If their sizes are different, Assertation
    error will be raised.

    Args:
        shape_kwargs(Dict[str, Optional[Tuple[int]]]): Expected shape.
            Not define the shape where the shape check is not needed.
            Also if the shape check for a part of axis is not needed, you can use None such as `x=(None, 1)`.

    Examples:

        .. code-block:: python

            @accepted_shapes(x=(3, 2), y=(1, 5), z=(None, 3))
            def dummy_function(x, y, z, non_shape_args=False):
                pass

            # Assertation error will be raised, x size is different
            dummy_function(x=np.zeros((3, 3)), y=np.zeros((1, 5)), z=np.zeros((3, 3)), non_shape_args=False)

            # Pass the decorator
            dummy_function(x=np.zeros((3, 2)), y=np.zeros((1, 5)), z=np.zeros((3, 3)), non_shape_args=False)

            # You can define the decorator in this way not to check the shape of z
            @accepted_shapes(x=(3, 2), y=(1, 5))
            def dummy_function(x, y, z, non_shape_args=False):
                pass
    """
    def accepted_shapes_wrapper(f: F) -> F:
        signature_f = signature(f)

        def wrapped_with_accepted_shapes(*args, **kwargs):
            binded_args = signature_f.bind(*args, **kwargs)
            _check_kwargs_shape(binded_args.arguments, shape_kwargs)
            return f(*args, **kwargs)

        return cast(F, wrapped_with_accepted_shapes)
    return accepted_shapes_wrapper


def _is_same_shape(actual_shape: Tuple[int], expected_shape: Tuple[int]) -> bool:
    if len(actual_shape) != len(expected_shape):
        return False
    return all([actual == expected or expected is None
                for actual, expected in zip(actual_shape, expected_shape)])


def _check_kwargs_shape(kwargs, expected_kwargs_shapes):
    for kw, expected_shape in expected_kwargs_shapes.items():
        assert kw in kwargs
        assert _is_same_shape(kwargs[kw].shape, expected_shape)
