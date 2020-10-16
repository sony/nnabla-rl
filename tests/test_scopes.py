import pytest

import nnabla_rl.scopes as scopes


class TestScopes(object):
    def test_eval_scope(self):
        assert not scopes.is_eval_scope()
        with scopes.eval_scope():
            assert scopes.is_eval_scope()
        assert not scopes.is_eval_scope()


if __name__ == "__main__":
    pytest.main()
