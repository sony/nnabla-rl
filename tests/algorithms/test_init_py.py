import pytest

from dataclasses import dataclass

from nnabla_rl.algorithm import Algorithm, AlgorithmParam
import nnabla_rl.algorithms as A


@dataclass
class SampleParam(AlgorithmParam):
    pass


class SampleAlgorithm(Algorithm):
    def __init__(self):
        pass


class TestInitPy(object):
    def setup_method(self, method):
        pass

    def test_register_algorithm(self):
        assert not A.is_registered(SampleAlgorithm, SampleParam)
        algorithm_num_before = len(A._ALGORITHMS)

        A.register_algorithm(SampleAlgorithm, SampleParam)

        assert A.is_registered(SampleAlgorithm, SampleParam)
        algorithm_num_after = len(A._ALGORITHMS)

        assert algorithm_num_before + 1 == algorithm_num_after

    def test_get_class_of(self):
        A.register_algorithm(SampleAlgorithm, SampleParam)

        (algorithm_class, param_class) = A.get_class_of(SampleAlgorithm.__name__)
        assert algorithm_class == SampleAlgorithm
        assert param_class == SampleParam

        with pytest.raises(KeyError):
            (algorithm_class, param_class) = A.get_class_of("UnknownClassName")


if __name__ == "__main__":
    pytest.main()
