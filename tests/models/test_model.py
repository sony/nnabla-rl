import pytest

from nnabla_rl.models.model import Model


class TestModel(object):
    def test_scope_name(self):
        scope_name = "test"
        model = Model(scope_name=scope_name)

        assert scope_name == model.scope_name

    def test_get_parameters(self):
        scope_name = "test"
        model = Model(scope_name=scope_name)

        assert len(model.get_parameters()) == 0


if __name__ == "__main__":
    pytest.main()
