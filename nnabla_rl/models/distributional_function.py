from abc import abstractmethod, ABCMeta

import numpy as np

import nnabla as nn

from nnabla_rl.models.model import Model
from nnabla_rl.models.q_function import QFunction

import nnabla.functions as NF
import nnabla_rl.functions as RF


class ValueDistributionFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name, num_actions, num_atoms, v_min, v_max):
        super(ValueDistributionFunction, self).__init__(scope_name)
        self._num_actions = num_actions
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max

    @abstractmethod
    def probabilities(self, s):
        raise NotImplementedError

    def argmax_q_from_probabilities(self, atom_probabilities):
        q_values = self.probabilities_to_q_values(atom_probabilities)
        return RF.argmax(q_values, axis=1)

    def probabilities_to_q_values(self, atom_probabilities):
        batch_size = atom_probabilities.shape[0]
        assert atom_probabilities.shape == (batch_size, self._num_actions, self._num_atoms)
        z = self._compute_z(self._num_atoms, self._v_min, self._v_max)
        z = RF.expand_dims(nn.Variable.from_numpy_array(z), axis=0)
        z = RF.expand_dims(z, axis=1)
        z = NF.broadcast(z, shape=(batch_size, self._num_actions, self._num_atoms))
        q_values = NF.sum(z * atom_probabilities, axis=2)
        assert q_values.shape == (batch_size, self._num_actions)
        return q_values

    def as_q_function(self) -> QFunction:
        class Wrapper(QFunction):
            def __init__(self, value_distribution_function):
                super(Wrapper, self).__init__(value_distribution_function.scope_name)
                self._value_distribution_function = value_distribution_function

            def q(self, s, a):
                q_values = self._value_distribution_function._state_to_q_values(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def max_q(self, s):
                q_values = self._value_distribution_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s):
                probabilities = self._value_distribution_function.probabilities(s)
                greedy_action = self._value_distribution_function.argmax_q_from_probabilities(probabilities)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s):
        probabilities = self.probabilities(s)
        return self.probabilities_to_q_values(probabilities)

    def _compute_z(self, num_atoms, v_min, v_max):
        delta_z = (v_max - v_min) / (num_atoms - 1)
        z = [v_min + i * delta_z for i in range(num_atoms)]
        return np.asarray(z)

    def _probabilities_of(self, probabilities, a):
        probabilities = NF.transpose(probabilities, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        probabilities = probabilities * one_hot
        probabilities = NF.sum(probabilities, axis=2)

        return probabilities

    def _to_one_hot(self, a):
        batch_size = a.shape[0]
        a = NF.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = NF.one_hot(a, (self._num_actions,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=(batch_size, self._num_atoms, self._num_actions))
        return one_hot


class QuantileDistributionFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name, num_actions, num_quantiles):
        super(QuantileDistributionFunction, self).__init__(scope_name)

        self._num_actions = num_actions
        self._num_quantiles = num_quantiles
        self._qj = 1 / num_quantiles

    @abstractmethod
    def quantiles(self, s):
        raise NotImplementedError

    def argmax_q_from_quantiles(self, quantiles):
        q_values = self.quantiles_to_q_values(quantiles)
        return RF.argmax(q_values, axis=1)

    def quantiles_to_q_values(self, quantiles):
        return NF.sum(quantiles * self._qj, axis=2)

    def as_q_function(self):
        class Wrapper(QFunction):
            def __init__(self, quantile_distribution_function):
                super(Wrapper, self).__init__(quantile_distribution_function.scope_name)
                self._quantile_distribution_function = quantile_distribution_function

            def q(self, s, a):
                q_values = self._quantile_distribution_function._state_to_q_values(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def max_q(self, s):
                q_values = self._quantile_distribution_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s):
                quantiles = self._quantile_distribution_function.quantiles(s)
                greedy_action = self._quantile_distribution_function.argmax_q_from_quantiles(quantiles)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s):
        quantiles = self.quantiles(s)
        return self.quantiles_to_q_values(quantiles)

    def _quantiles_of(self, quantiles, a):
        batch_size = quantiles.shape[0]
        quantiles = NF.transpose(quantiles, axes=(0, 2, 1))
        one_hot = self._to_one_hot(a)
        quantiles = quantiles * one_hot
        quantiles = NF.sum(quantiles, axis=2)
        assert quantiles.shape == (batch_size, self._num_quantiles)

        return quantiles

    def _to_one_hot(self, a):
        batch_size = a.shape[0]
        a = NF.reshape(a, (-1, 1))
        assert a.shape[0] == batch_size
        one_hot = NF.one_hot(a, (self._num_actions,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=(batch_size, self._num_quantiles, self._num_actions))
        return one_hot


def risk_neutral_measure(tau):
    return tau


class StateActionQuantileFunction(Model, metaclass=ABCMeta):
    def __init__(self, scope_name, num_actions, K, risk_measure_function=risk_neutral_measure):
        super(StateActionQuantileFunction, self).__init__(scope_name)

        self._num_actions = num_actions
        self._K = K
        self._risk_measure_function = risk_measure_function

    @abstractmethod
    def quantiles(self, s, tau):
        pass

    def argmax_q_from_quantiles(self, quantiles):
        q_values = self.quantiles_to_q_values(quantiles)
        return RF.argmax(q_values, axis=1)

    def quantiles_to_q_values(self, quantiles):
        quantiles = NF.transpose(quantiles, axes=(0, 2, 1))
        q_values = NF.mean(quantiles, axis=2)
        return q_values

    def as_q_function(self) -> QFunction:
        class Wrapper(QFunction):
            def __init__(self, quantile_function):
                super(Wrapper, self).__init__(quantile_function.scope_name)
                self._quantile_function = quantile_function

            def q(self, s, a):
                q_values = self._quantile_function._state_to_q_values(s)
                one_hot = NF.one_hot(NF.reshape(a, (-1, 1), inplace=False), (q_values.shape[1],))
                q_value = NF.sum(q_values * one_hot, axis=1, keepdims=True)  # get q value of a
                return q_value

            def max_q(self, s):
                q_values = self._quantile_function._state_to_q_values(s)
                return NF.max(q_values, axis=1, keepdims=True)

            def argmax_q(self, s):
                batch_size = s.shape[0]
                tau = self._quantile_function._sample_risk_measured_tau(shape=(batch_size, self._quantile_function._K))
                quantiles = self._quantile_function.quantiles(s, tau)
                greedy_action = self._quantile_function.argmax_q_from_quantiles(quantiles)
                return greedy_action

        return Wrapper(self)

    def _state_to_q_values(self, s):
        tau = self._sample_risk_measured_tau(shape=(1, self._K))
        quantiles = self.quantiles(s, tau)
        return self.quantiles_to_q_values(quantiles)

    def _quantiles_of(self, quantiles, a):
        one_hot = self._to_one_hot(a, shape=quantiles.shape)
        quantiles = quantiles * one_hot
        quantiles = NF.sum(quantiles, axis=2)
        assert len(quantiles.shape) == 2

        return quantiles

    def _to_one_hot(self, a, shape):
        a = NF.reshape(a, (-1, 1))
        one_hot = NF.one_hot(a, (self._num_actions,))
        one_hot = RF.expand_dims(one_hot, axis=1)
        one_hot = NF.broadcast(one_hot, shape=shape)
        return one_hot

    def _sample_tau(self, shape=None):
        return NF.rand(low=0.0, high=1.0, shape=shape)

    def _sample_risk_measured_tau(self, shape):
        tau = self._sample_tau(shape)
        return self._risk_measure_function(tau)
