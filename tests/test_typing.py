# Copyright 2023 Sony Group Corporation.
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

import numpy as np
import pytest

from nnabla_rl.typing import accepted_shapes


class TestTyping():
    def test_accepted_shapes_call_with_args(self):
        @accepted_shapes(x=(3, 5), u=(2, 4))
        def dummy_function(x, u):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 2)), np.zeros((2, 4)))

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((3, 5)), np.zeros((4, 3)))

        dummy_function(np.zeros((3, 5)), np.zeros((2, 4)))

    def test_accepted_shapes_call_with_kwargs(self):
        @accepted_shapes(x=(3, 5), u=(2, 4))
        def dummy_function(x=np.zeros((3, 5)), u=np.zeros((2, 4))):
            pass

        with pytest.raises(AssertionError):
            dummy_function(x=np.zeros((4, 2)), u=np.zeros((2, 4)))

        with pytest.raises(AssertionError):
            dummy_function(x=np.zeros((3, 5)), u=np.zeros((4, 3)))

        dummy_function(x=np.zeros((3, 5)), u=np.zeros((2, 4)))

    def test_accepted_shapes_call_with_args_and_kwargs(self):
        @accepted_shapes(x=(4, 3), u=(2, 1))
        def dummy_function(x, u=np.ones((2, 1))):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((3, 2)), u=np.zeros((2, 1)))

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 3)), u=np.zeros((2, 4)))

        dummy_function(np.zeros((4, 3)), u=np.zeros((2, 1)))

    def test_accepted_shapes_decorator_has_None_shape_part(self):
        @accepted_shapes(x=(None, 3), u=(2, None))
        def dummy_function(x, u=np.ones((2, 1))):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((3, 2)), u=np.zeros((2, 1)))

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 3)), u=np.zeros((3, 4)))

        dummy_function(np.zeros((10, 3)), u=np.zeros((2, 1)))

    def test_accepted_shapes_call_with_non_shapes_kwargs(self):
        @accepted_shapes(x=(4, 3), u=(2, 1))
        def dummy_function(x, u=np.ones((2, 1)), batched=True):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 2)), u=np.zeros((2, 1)), batched=True)

        dummy_function(np.zeros((4, 3)), u=np.zeros((2, 1)), batched=True)

    def test_accepted_shapes_call_with_args_as_kwargs(self):
        @accepted_shapes(x=(4, 3), u=(2, 1))
        def dummy_function(x, u=np.ones((2, 1)), batched=True):
            pass

        with pytest.raises(AssertionError):
            dummy_function(x=np.zeros((4, 2)), u=np.zeros((2, 1)), batched=True)

        dummy_function(x=np.zeros((4, 3)), u=np.zeros((2, 1)), batched=True)

    def test_accepted_shapes_decorator_has_invalid_args(self):
        with pytest.raises(TypeError):
            @accepted_shapes((4, 3), u=(2, 1))
            def dummy_function(x, u=np.ones((2, 1)), batched=True):
                pass

    def test_accepted_shapes_decorator_has_less_args_than_function_args(self):
        @accepted_shapes(x=(4, 3))
        def dummy_function(x, u=np.ones((2, 1)), batched=True):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((5, 3)), u=np.zeros((2, 1)), batched=True)

        dummy_function(np.zeros((4, 3)), u=np.zeros((2, 1)), batched=True)

    def test_accepted_shapes_decorator_has_more_args_than_function_args(self):
        @accepted_shapes(x=(4, 3), u=(2, 1), t=(3, 4))
        def dummy_function(x, u=np.ones((2, 1)), batched=True):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 3)), u=np.zeros((2, 1)), batched=True)

    def test_accepted_shapes_decorator_has_wrong_args_with_function_args(self):
        @accepted_shapes(x=(4, 3), uu=(2, 1))
        def dummy_function(x, u=np.ones((2, 1)), batched=True):
            pass

        with pytest.raises(AssertionError):
            dummy_function(np.zeros((4, 3)), u=np.zeros((2, 1)), batched=True)

    def test_accepted_shapes_decorator_has_no_args(self):
        @accepted_shapes()
        def dummy_function(x, u):
            pass

        dummy_function(np.zeros((4, 3)), np.zeros((2, 1)))

    def test_accepted_shapes_call_with_different_kwargs_order(self):
        @accepted_shapes(x=(3, 5), u=(2, 4))
        def dummy_function(x=np.zeros((3, 5)), u=np.zeros((2, 4))):
            pass

        with pytest.raises(AssertionError):
            dummy_function(u=np.zeros((3, 5)), x=np.zeros((2, 4)))

        dummy_function(u=np.zeros((2, 4)), x=np.zeros((3, 5)))


if __name__ == '__main__':
    pytest.main()
