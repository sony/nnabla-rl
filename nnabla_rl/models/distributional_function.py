from abc import abstractmethod, ABCMeta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import nnabla as nn

from nnabla_rl.models.model import Model
from nnabla_rl.models.q_function import QFunction

import nnabla.functions as NF
import nnabla_rl.functions as RF


class ValueDistributionFunction(Model, metaclass=ABCMeta):

    _n_action: int
    _n_atom: int
    _v_min: float
    _v_max: float
    _z: nn.Variable

    def __init__(self, scope_name: str, n_action: int, n_atom: int, v_min: float, v_max: float):
        super(ValueDistributionFunction, self).__init__(scope_name)
        self._n_action = n_action
        self._n_atom = n_atom
        self._v_min = v_min
        self._v_max = v_max
        # precompute atoms
        self._z = self._compute_z(n_atom, v_min, v_max)
        self._z.persistent = True

    def __deepcopy__(self, memodict: Dict[Any, Any] = {}):
        # nn.Variable cannot be deepcopied directly
        return self.__class__(self._scope_name, self._n_action, self._n_atom, self._v_min, self._v_max)

    @abstractmethod
    def probabilities(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def argmax_q_from_probabilities(self, atom_probabilities: nn.Variable) -> nn.Variable:
        q_values = self.probabilities_to_q_values(atom_probabilities)
        return RF.argmax(q_values, axis=1, keepdims=True)

    def probabilities_to_q_values(self, atom_probabilities: nn.Variable) -> nn.Variable:
        batch_size = atom_probabilities.shape[0]
        assert atom_probabilities.shape == (batch_size, self._n_action, self._n_atom)
        z = RF.expand_dims(self._z, axis=0)
        z = RF.expand_dims(z, axis=1)
        z = NF.broadcast(z, shape=(batch_size, self._n_action, self._n_atom))
        q_values = NF.sum(z * atom_probabilities, axis=2)
        assert q_values.shape == (batch_size, self._n_action)
        return q_values

    def as_q_function(self) -> QFunction:
        class Wrapper(QFunction):

            _value_distribution_function: 'ValueDistributionFunction'

            def __init__(self, value_distribution_function: 'ValueDistributionFunction'):
                super(Wrapper, self).__init__(value_distribution_function.scope_name)
                self._value_distribution_function = value_distribution_function

            def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
                q_values = self._value_distribution_function._state_to_q_values(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def max_q(self, s: nn.Variable) -> nn.Variable:
                q_values = self._value_distribution_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s: nn.Variable) -> nn.Variable:
                probabilities = self._value_distribution_function.probabilities(s)
                greedy_action = self._value_distribution_function.argmax_q_from_probabilities(probabilities)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s: nn.Variable) -> nn.Variable:
        probabilities = self.probabilities(s)
        return self.probabilities_to_q_values(probabilities)

    def _compute_z(self, n_atom: int, v_min: float, v_max: float) -> nn.Variable:
        delta_z = (v_max - v_min) / (n_atom - 1)
        z = nn.Variable.from_numpy_array(np.asarray([v_min + i * delta_z for i in range(n_atom)]))
        return z

    def _probabilities_of(self, probabilities: nn.Variable, a: nn.Variable) -> nn.Variable:
        probabilities = NF.transpose(probabilities, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        probabilities = probabilities * one_hot
        probabilities = NF.sum(probabilities, axis=2)

        return probabilities

    def _to_one_hot(self, a: nn.Variable) -> nn.Variable:
        batch_size = a.shape[0]
        a = NF.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = NF.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=(batch_size, self._n_atom, self._n_action))
        return one_hot


class QuantileDistributionFunction(Model, metaclass=ABCMeta):

    _n_action: int
    _n_quantile: int
    _qj: float

    def __init__(self, scope_name: str, n_action: int, n_quantile: int):
        super(QuantileDistributionFunction, self).__init__(scope_name)
        self._n_action = n_action
        self._n_quantile = n_quantile
        self._qj = 1 / n_quantile

    @abstractmethod
    def quantiles(self, s: nn.Variable) -> nn.Variable:
        raise NotImplementedError

    def argmax_q_from_quantiles(self, quantiles: nn.Variable) -> nn.Variable:
        q_values = self.quantiles_to_q_values(quantiles)
        return RF.argmax(q_values, axis=1, keepdims=True)

    def quantiles_to_q_values(self, quantiles: nn.Variable) -> nn.Variable:
        return NF.sum(quantiles * self._qj, axis=2)

    def as_q_function(self) -> QFunction:
        class Wrapper(QFunction):

            _quantile_distribution_function: 'QuantileDistributionFunction'

            def __init__(self, quantile_distribution_function: 'QuantileDistributionFunction'):
                super(Wrapper, self).__init__(quantile_distribution_function.scope_name)
                self._quantile_distribution_function = quantile_distribution_function

            def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
                q_values = self._quantile_distribution_function._state_to_q_values(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def max_q(self, s: nn.Variable) -> nn.Variable:
                q_values = self._quantile_distribution_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s: nn.Variable) -> nn.Variable:
                quantiles = self._quantile_distribution_function.quantiles(s)
                greedy_action = self._quantile_distribution_function.argmax_q_from_quantiles(quantiles)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s: nn.Variable) -> nn.Variable:
        quantiles = self.quantiles(s)
        return self.quantiles_to_q_values(quantiles)

    def _quantiles_of(self, quantiles: nn.Variable, a: nn.Variable) -> nn.Variable:
        batch_size = quantiles.shape[0]
        quantiles = NF.transpose(quantiles, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        quantiles = quantiles * one_hot
        quantiles = NF.sum(quantiles, axis=2)
        assert quantiles.shape == (batch_size, self._n_quantile)

        return quantiles

    def _to_one_hot(self, a: nn.Variable) -> nn.Variable:
        batch_size = a.shape[0]
        a = NF.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = NF.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=(batch_size, self._n_quantile, self._n_action))
        return one_hot


def risk_neutral_measure(tau: nn.Variable) -> nn.Variable:
    return tau


class StateActionQuantileFunction(Model, metaclass=ABCMeta):

    _n_action: int
    _k: float
    _risk_measure_function: Callable[[nn.Variable], nn.Variable]

    def __init__(self,
                 scope_name: str,
                 n_action: int,
                 K: float,
                 risk_measure_function: Callable[[nn.Variable], nn.Variable] = risk_neutral_measure):
        super(StateActionQuantileFunction, self).__init__(scope_name)
        self._n_action = n_action
        self._K = K
        self._risk_measure_function = risk_measure_function

    @abstractmethod
    def quantiles(self, s: nn.Variable, tau: nn.Variable) -> nn.Variable:
        pass

    def argmax_q_from_quantiles(self, quantiles: nn.Variable) -> nn.Variable:
        q_values = self.quantiles_to_q_values(quantiles)
        return RF.argmax(q_values, axis=1, keepdims=True)

    def quantiles_to_q_values(self, quantiles: nn.Variable) -> nn.Variable:
        quantiles = NF.transpose(quantiles, axes=(0, 2, 1))
        q_values = NF.mean(quantiles, axis=2)
        return q_values

    def as_q_function(self) -> QFunction:
        class Wrapper(QFunction):

            _quantile_function: 'StateActionQuantileFunction'

            def __init__(self, quantile_function: 'StateActionQuantileFunction'):
                super(Wrapper, self).__init__(quantile_function.scope_name)
                self._quantile_function = quantile_function

            def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
                q_values = self.all_q(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def all_q(self, s: nn.Variable) -> nn.Variable:
                return self._quantile_function._state_to_q_values(s)

            def max_q(self, s: nn.Variable) -> nn.Variable:
                q_values = self._quantile_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s: nn.Variable) -> nn.Variable:
                batch_size = s.shape[0]
                tau = self._quantile_function._sample_risk_measured_tau(shape=(batch_size, self._quantile_function._K))
                quantiles = self._quantile_function.quantiles(s, tau)
                greedy_action = self._quantile_function.argmax_q_from_quantiles(quantiles)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s: nn.Variable) -> nn.Variable:
        tau = self._sample_risk_measured_tau(shape=(1, self._K))
        quantiles = self.quantiles(s, tau)
        return self.quantiles_to_q_values(quantiles)

    def _quantiles_of(self, quantiles: nn.Variable, a: nn.Variable) -> nn.Variable:
        one_hot = self._to_one_hot(a, shape=quantiles.shape)
        quantiles = quantiles * one_hot
        quantiles = NF.sum(quantiles, axis=2)
        assert len(quantiles.shape) == 2

        return quantiles

    def _to_one_hot(self, a: nn.Variable, shape: nn.Variable) -> nn.Variable:
        a = NF.reshape(a, (-1, 1))
        one_hot = NF.one_hot(a, (self._n_action,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=shape)
        return one_hot

    def _sample_tau(self, shape: Optional[List] = None) -> nn.Variable:
        return RF.rand(low=0.0, high=1.0, shape=shape)

    def _sample_risk_measured_tau(self, shape: Optional[List]) -> nn.Variable:
        tau = self._sample_tau(shape)
        return self._risk_measure_function(tau)
