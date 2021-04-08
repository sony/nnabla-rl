# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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

import nnabla_rl.scopes as scopes


class TestScopes(object):
    def test_eval_scope(self):
        assert not scopes.is_eval_scope()
        with scopes.eval_scope():
            assert scopes.is_eval_scope()
        assert not scopes.is_eval_scope()

    def test_nested_eval_scope(self):
        assert not scopes.is_eval_scope()
        with scopes.eval_scope():
            assert scopes.is_eval_scope()
            with scopes.eval_scope():
                assert scopes.is_eval_scope()
            assert scopes.is_eval_scope()
        assert not scopes.is_eval_scope()


if __name__ == "__main__":
    pytest.main()
