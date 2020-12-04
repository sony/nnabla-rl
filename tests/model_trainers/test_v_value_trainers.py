import pytest

import nnabla as nn


class TestTargetValueBasedVFunctionTrainer(object):
    def setup_method(self, method):
        nn.clear_parameters()


if __name__ == "__main__":
    pytest.main()
