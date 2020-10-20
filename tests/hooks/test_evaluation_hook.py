from unittest import mock

import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.hooks import EvaluationHook


class TestEvaluationHook():
    def test_call(self):
        dummy_env = E.DummyContinuous()

        dummy_algorithm = A.Dummy(dummy_env)

        mock_evaluator = mock.MagicMock()
        mock_evaluator.return_value = [0]

        mock_writer = mock.MagicMock()

        hook = EvaluationHook(dummy_env,
                              evaluator=mock_evaluator,
                              writer=mock_writer)

        hook(dummy_algorithm)

        inputs = (dummy_algorithm, dummy_env)

        mock_evaluator.assert_called_once_with(*inputs)
        hook._writer.write_scalar.assert_called_once()
